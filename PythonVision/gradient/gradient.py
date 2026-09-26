#!/usr/bin/env python3
"""Pyramidal, masked ground-plane registration for the CDFR simulator.

Localizes the robot by maximizing normalized cross-correlation (NCC) between the
rectified ground patch and a fixed top-down map, coarse-to-fine over an image
pyramid, from several starting poses. Optionally fuses an ArUco tag's absolute
pose. No output image is needed for localization; import `Localizer` for a live
camera.

Run with the shared harness:

    ../.venv/bin/python ../script.py gradient
    ../.venv/bin/python ../script.py gradient capture_x.png --no-markers

------------------------------------------------------------------------------
COMMENTAIRES EN FRANÇAIS (ajoutés a posteriori, ne changent rien au code) :

Ce script implémente un système de LOCALISATION VISUELLE pour un robot de
coupe de robotique (terrain 2 m x 3 m, type CDFR/Eurobot). Le principe :

  1. On dispose d'une image de référence du terrain vue du dessus (en N&B).
  2. Une caméra embarquée sur le robot regarde le sol en biais (45°).
  3. On "redresse" (rectifie) la photo pour obtenir une vue du dessus.
  4. On cherche, par optimisation, la pose (x, y, yaw) du robot qui fait
     le mieux correspondre ce patch redressé à l'image de référence, via
     une corrélation croisée normalisée (NCC) calculée à plusieurs échelles
     (pyramide grossier -> fin), en partant de plusieurs points de départ.
  5. En option, des marqueurs ArUco posés sur le terrain donnent une pose
     absolue qui vient recaler ou fusionner avec le résultat photométrique.

Le code ne modifie jamais l'image ; aucune sortie graphique n'est nécessaire
pour la localisation elle-même (elle sert surtout de simulateur/évaluateur).

NOTE DE PORTAGE : ce module reste volontairement autonome. Il redéfinit sa
propre homographie et son propre `ground_bounds` (boucle 33x33, là où
`common.geometry` échantillonne 65x65) parce que la rectification est ici au
cœur d'un optimiseur et que les deux variantes ont été mesurées séparément.
`TAG_FIELD_MM`, `FIELD_WIDTH_MM`, `FIELD_HEIGHT_MM` et les constantes caméra
viennent en revanche de `common.geometry`, pour que le terrain et le montage
ne puissent pas diverger entre les trois algorithmes.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np

# Both the harness (which imports this module by path) and a direct
# `python gradient.py` need PythonVision/ importable for `common`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, pose  # noqa: E402
from common.geometry import (  # noqa: E402
    CAMERA_HEIGHT_MM,
    CAMERA_PATCH_CENTRE_MM,
    CAMERA_PITCH_DEG,
    CAMERA_VFOV_DEG,
    FIELD_HEIGHT_MM,
    FIELD_WIDTH_MM,
    REFERENCE_PAD_VALUE,
    TAG_FIELD_MM,
)

HERE = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# CONSTANTES PHYSIQUES ET GÉOMÉTRIQUES
# -----------------------------------------------------------------------------
# Ces valeurs décrivent le terrain et la caméra. Elles définissent le modèle
# géométrique fixe utilisé pour convertir un pixel de la photo en une position
# réelle sur le sol (en millimètres). Le terrain fait 2 m x 3 m (dimensions
# classiques d'un terrain de Coupe de France de Robotique / Eurobot).
#
# `FIELD_WIDTH_MM`, `FIELD_HEIGHT_MM`, `CAMERA_VFOV_DEG`, `CAMERA_PITCH_DEG`,
# `CAMERA_HEIGHT_MM`, `CAMERA_PATCH_CENTRE_MM`, `REFERENCE_PAD_VALUE` et
# `TAG_FIELD_MM` sont importés de `common.geometry` : ils étaient identiques
# dans les trois algorithmes, donc ils y vivent désormais.
#
# Distance (en mm), le long de l'axe de visée, entre la caméra et un point de
# référence appelé "centre" du patch observé. Le code raisonne en interne sur
# la pose de ce "centre" plutôt que sur la pose de la caméra elle-même (voir
# camera_to_centre / centre_to_camera plus bas) car c'est ce point qui reste
# le plus stable/central dans le patch redressé pendant l'optimisation.
#
# Position connue (en mm, repère terrain) des 4 marqueurs ArUco fixés au sol,
# indexés par leur identifiant de tag. Utilisés pour la localisation absolue.
# (voir TAG_FIELD_MM dans common.geometry)

# Expression régulière qui extrait x, y, yaw depuis un nom de fichier du type
# "capture_x100.0_y-200.5_yaw30.png". C'est la "vérité terrain" utilisée pour
# évaluer la précision de la localisation dans run().
# (voir common.pose.POSE_IN_NAME)


def read_gray(path: Path) -> np.ndarray:
    """Charge une image depuis le disque en niveaux de gris (8 bits)."""
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def parse_pose(path: Path) -> tuple[float, float, float]:
    """Extrait la pose (x, y, yaw) codée dans le nom de fichier (vérité terrain)."""
    truth = pose.parse_pose_filename(str(path))
    if truth is None:
        raise ValueError(f"No x/y/yaw pose in {path.name}")
    return truth


def wrap_deg(angle: float) -> float:
    """Ramène un angle quelconque (en degrés) dans l'intervalle [-180, 180[."""
    return pose.wrap_deg(angle)


def camera_to_centre(pose_: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convertit une pose CAMÉRA (x, y, yaw) en pose du point "centre" du patch.

    Le point "centre" est simplement décalé de CAMERA_PATCH_CENTRE_MM devant
    la caméra, dans la direction pointée par le yaw. L'orientation (yaw) est
    inchangée : seule la translation est appliquée.
    """
    x, y, yaw = pose_
    rad = math.radians(yaw)
    return (x + CAMERA_PATCH_CENTRE_MM * math.cos(rad),
            y + CAMERA_PATCH_CENTRE_MM * math.sin(rad), yaw)


def centre_to_camera(pose_: tuple[float, float, float]) -> tuple[float, float, float]:
    """Opération inverse de camera_to_centre : "centre" -> pose caméra."""
    x, y, yaw = pose_
    rad = math.radians(yaw)
    return (x - CAMERA_PATCH_CENTRE_MM * math.cos(rad),
            y - CAMERA_PATCH_CENTRE_MM * math.sin(rad), wrap_deg(yaw))


def capture_homography(width: int, height: int) -> np.ndarray:
    """Image pixel -> local ground (right, forward), in millimetres.

    This is the fixed 45 degree ground-plane model of the original script.
    Filename pitch is deliberately ignored.

    --- FR ---
    Construit l'HOMOGRAPHIE (matrice 3x3) qui envoie un pixel homogène
    (u, v, 1) de l'image capturée vers un point homogène (X, Z, w) tel que
    (X/w, Z/w) donne la position sur le sol, en mm, dans le repère local du
    robot (X = vers la droite, Z = vers l'avant). C'est un modèle purement
    géométrique (pas de calibration par damier) : on suppose un pitch fixe
    de 45° et un champ de vision vertical fixe de 70°, quelle que soit la
    capture (le pitch éventuellement présent dans le nom de fichier est
    volontairement ignoré, comme le précise le commentaire d'origine).

    Étapes :
      1. Calcule la focale (en pixels) à partir du champ de vision vertical.
      2. Construit la matrice caméra inverse K^-1 (pixel -> rayon caméra).
      3. Construit une base orthonormée (right, up, forward) correspondant
         à l'orientation de la caméra une fois inclinée de CAMERA_PITCH_DEG.
      4. Combine cette base avec K^-1 pour obtenir, pour chaque pixel, un
         rayon 3D exprimé dans le repère monde (aligné avec le sol).
      5. Sachant que la caméra est à une hauteur connue au-dessus du plan du
         sol (Y = -CAMERA_HEIGHT_MM), calcule où chaque rayon intersecte ce
         plan : c'est une simple mise à l'échelle par la composante de
         profondeur (rays[1], la coordonnée verticale du rayon).
      6. Convertit le résultat en mm (les calculs intermédiaires sont en
         mètres) grâce à la matrice diagonale finale.
    """
    focal = (height / 2.0) / math.tan(math.radians(CAMERA_VFOV_DEG / 2.0))
    inv_k = np.array([[1 / focal, 0, -width / (2 * focal)],
                      [0, -1 / focal, height / (2 * focal)],
                      [0, 0, 1]], dtype=np.float64)
    pitch = math.radians(CAMERA_PITCH_DEG)
    # "forward" est le vecteur de visée de la caméra, incliné de `pitch` vers
    # le bas (composante Y négative = vers le bas, Z négative = vers l'avant).
    forward = np.array([0.0, -math.sin(pitch), -math.cos(pitch)])
    # "right" est perpendiculaire à "forward" et à la verticale monde (0,1,0).
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    # "up" complète la base orthonormée (produit vectoriel right x forward).
    up = np.cross(right, forward)
    # Chaque colonne de `rays` est le rayon 3D (dans le repère monde) associé
    # à un pixel donné, une fois qu'on applique K^-1 puis le changement de
    # base (right, up, forward).
    rays = np.column_stack((right, up, forward)) @ inv_k
    depth = rays[1]  # composante verticale du rayon (sert de dénominateur).
    # Intersection rayon/plan du sol : on met à l'échelle chaque rayon par
    # (hauteur caméra / composante verticale) pour atteindre Y = -hauteur.
    # Le résultat, exprimé en mètres, donne (X, Z, w) homogènes.
    h_metres = np.vstack((-CAMERA_HEIGHT_MM / 1000.0 * rays[0],
                          -CAMERA_HEIGHT_MM / 1000.0 * rays[2], depth))
    # On repasse en millimètres (les deux premières lignes sont X, Z en m).
    return np.diag([1000.0, 1000.0, 1.0]) @ h_metres


def ground_bounds(homography: np.ndarray, size: tuple[int, int]) -> tuple[float, ...]:
    """Calcule l'empreinte au sol (rectangle englobant, en mm) vue par la caméra.

    Utilise l'homographie ci-dessus pour projeter une grille régulière de
    points de l'image (33x33) sur le sol, puis en déduit le rectangle
    englobant (avec une petite marge de 2 %). Les points qui partent vers
    l'horizon (projected[2] >= 0, c'est-à-dire un rayon "vers le haut", qui
    ne coupe jamais le plan du sol devant la caméra) sont exclus : c'est ce
    qui permet de gérer le cas où une partie de l'image ne voit pas le sol.
    """
    width, height = size
    # The horizon can make a footprint invalid; sample the image grid once.
    xs = np.linspace(0, width - 1, 33)
    ys = np.linspace(0, height - 1, 33)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)), axis=0)
    projected = homography @ points
    valid = projected[2] < -1e-9  # ne garder que les rayons qui touchent le sol.
    if not np.any(valid):
        raise ValueError("Camera view has no ground-plane intersection")
    ground = projected[:2, valid] / projected[2, valid]  # normalisation homogène.
    x0, z0 = ground.min(axis=1)
    x1, z1 = ground.max(axis=1)
    pad = 0.02 * max(x1 - x0, z1 - z0)
    return (x0 - pad, z0 - pad, x1 + pad, z1 + pad)


def valid_region_mask(warped_camera_coverage: np.ndarray) -> np.ndarray:
    """Only pixels fully supported by a camera pixel, including black content.

    --- FR ---
    Après un warp perspective, un pixel de destination peut être une
    interpolation partielle de plusieurs pixels source (ou n'avoir aucune
    source si le point projeté sort de l'image). Pour distinguer un vrai
    contenu noir (valeur 0 mais valide) d'un pixel simplement non couvert,
    on warp un masque de couverture 100 % blanc (255) séparément : seuls les
    pixels de destination qui valent encore exactement 255 après le warp ont
    été entièrement "vus" par un pixel source. Une petite érosion 3x3 exclut
    en plus les pixels de bord, potentiellement contaminés par interpolation.
    """
    strict = cv.compare(warped_camera_coverage, 255, cv.CMP_EQ)
    return cv.erode(strict, np.ones((3, 3), np.uint8))


@dataclass
class Level:
    """Représente un niveau de la pyramide multi-résolution, pour UNE capture.

    reference       : image de référence (terrain) à ce niveau, en float32.
    field_mask      : masque des pixels valides de la référence (érodé).
    mm_per_px       : résolution de ce niveau (mm par pixel).
    patch           : image caméra rectifiée (vue du dessus) à ce niveau, float32.
    patch_squared   : carré pixel à pixel de `patch` (précalculé pour la variance).
    patch_mask      : masque des pixels valides du patch rectifié.
    valid_count     : nombre de pixels valides dans patch_mask.
    radius_mm       : rayon (demi-diagonale) du patch, en mm (sert à convertir
                      un pas angulaire en un déplacement de pixels équivalent).
    centre_uv       : coordonnées (u, v), dans le patch, du point "centre"
                      utilisé comme origine des poses (voir _matrix).
    """
    reference: np.ndarray
    field_mask: np.ndarray
    mm_per_px: float
    patch: np.ndarray
    patch_squared: np.ndarray
    patch_mask: np.ndarray
    valid_count: int
    radius_mm: float
    centre_uv: tuple[float, float]


@dataclass
class Result:
    """Résultat renvoyé par Localizer.locate().

    x_mm, y_mm, yaw_deg : pose CAMÉRA estimée (ou pose initiale si échec).
    loss                : perte finale (photométrique, ou fusionnée avec le
                          marqueur si un tag a été utilisé).
    correlation         : corrélation croisée normalisée (NCC) au niveau final.
    valid_fraction      : fraction de pixels valides utilisés dans le score.
    elapsed_ms          : temps de calcul total, en millisecondes.
    evaluations         : nombre total d'appels à la fonction de coût.
    status              : "ok", "marker_fused", "low_confidence" ou
                          "insufficient_overlap_or_texture".
    marker_count        : nombre de marqueurs ArUco détectés et exploitables.
    objective           : valeur brute de la fonction objectif optimisée
                          (peut différer de `loss`/`correlation` en mode fusion).
    """
    x_mm: float
    y_mm: float
    yaw_deg: float
    loss: float
    correlation: float
    valid_fraction: float
    elapsed_ms: float
    evaluations: int
    status: str
    marker_count: int
    objective: float


class Localizer:
    """Objet réutilisable : encapsule la pyramide de référence et l'état du
    détecteur de marqueurs, pour localiser efficacement de nombreuses captures
    successives (typiquement en flux vidéo temps réel sur le robot).
    """

    def __init__(self, reference: np.ndarray, max_reference_side: int = 384,
                 levels: int = 3, iterations: int = 14,
                 max_shift_mm: float = 600.0, max_yaw_deg: float = 45.0,
                 use_markers: bool = True, source_top_fraction: float = 0.15,
                 multistart: int = 5, prior_xy_sigma_mm: float = 100.0,
                 prior_yaw_sigma_deg: float = 5.0):
        # --- FR : validation des paramètres --------------------------------
        # Tous ces contrôles lèvent une ValueError explicite si un paramètre
        # d'entrée est incohérent (image couleur au lieu de niveaux de gris,
        # tailles/itérations négatives, fractions hors intervalle, etc.).
        if reference.ndim != 2 or not 0 < max_reference_side or not 0 < levels:
            raise ValueError("Expected a grayscale reference, positive side and levels")
        if max_shift_mm <= 0 or max_yaw_deg <= 0 or iterations <= 0:
            raise ValueError("Search bounds and iterations must be positive")
        if not 0 <= source_top_fraction < 0.8:
            raise ValueError("source_top_fraction must be in [0, 0.8)")
        if multistart < 1 or prior_xy_sigma_mm <= 0 or prior_yaw_sigma_deg <= 0:
            raise ValueError("multistart and prior sigmas must be positive")

        # --- FR : construction de la pyramide de références ----------------
        # On redimensionne d'abord l'image de référence pour que son plus
        # grand côté ne dépasse pas `max_reference_side` (niveau le plus fin),
        # puis on construit les niveaux plus grossiers par sous-échantillonnage
        # gaussien successif (cv.pyrDown, qui floute avant de diviser par 2 —
        # ce qui évite l'aliasing). `self.references` est ensuite inversé pour
        # aller du plus grossier au plus fin (ordre utilisé pendant `locate`).
        ratio = min(1.0, max_reference_side / max(reference.shape))
        finest_size = (max(1, round(reference.shape[1] * ratio)),
                       max(1, round(reference.shape[0] * ratio)))
        finest = cv.resize(reference, finest_size, interpolation=cv.INTER_AREA)
        images = [finest]
        for _ in range(levels - 1):
            images.append(cv.pyrDown(images[-1]))
        self.references = list(reversed(images))  # grossier -> fin

        # --- FR : pré-calcul, pour chaque niveau, du masque et de mm/px ----
        self.reference_levels = []
        for ref in self.references:
            # La résolution (mm/px) de ce niveau est déduite de sa largeur en
            # pixels par rapport à la largeur réelle du terrain (2000 mm).
            mm_per_px = FIELD_WIDTH_MM / ref.shape[1]
            # Vérifie que l'image de référence respecte bien le ratio 2000x3000
            # (à 2 % près) : sinon l'échelle mm/px serait fausse verticalement.
            if abs(mm_per_px - FIELD_HEIGHT_MM / ref.shape[0]) > 0.02 * mm_per_px:
                raise ValueError("Reference aspect ratio does not match 2000x3000 mm")
            # Masque "plein cadre" érodé de 1 pixel : exclut le bord de la
            # référence, où un warp pourrait introduire des artefacts.
            field_mask = cv.erode(np.full(ref.shape, 255, np.uint8),
                                  np.ones((3, 3), np.uint8),
                                  borderType=cv.BORDER_CONSTANT, borderValue=0)
            self.reference_levels.append((ref.astype(np.float32), field_mask,
                                          mm_per_px))

        # Cache des transformations de rectification (indexé par taille de
        # capture + résolution + source_top_fraction), pour éviter de
        # recalculer l'homographie et l'empreinte au sol à chaque frame.
        self._warp_cache: dict[tuple, tuple[np.ndarray, tuple[int, int], np.ndarray,
                                           tuple[float, float]]] = {}
        self.iterations = iterations
        self.max_shift_mm = max_shift_mm
        self.max_yaw_deg = max_yaw_deg
        self.use_markers = use_markers
        self.source_top_fraction = source_top_fraction
        self.multistart = multistart
        self.prior_xy_sigma_mm = prior_xy_sigma_mm
        self.prior_yaw_sigma_deg = prior_yaw_sigma_deg

        # --- FR : initialisation optionnelle du détecteur ArUco -------------
        self.detector = None
        if use_markers:
            if not hasattr(cv, "aruco") or not hasattr(cv.aruco, "ArucoDetector"):
                raise RuntimeError("cv.aruco.ArucoDetector unavailable; install OpenCV contrib or use --no-markers")
            params = cv.aruco.DetectorParameters()
            # Plage de tailles de fenêtre pour le seuillage adaptatif : aide
            # à détecter des tags de tailles variées dans l'image brute.
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 12
            # Raffinement sub-pixellique des coins détectés, pour plus de
            # précision géométrique (important car on en déduit une pose).
            params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
            dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
            self.detector = cv.aruco.ArucoDetector(dictionary, params)

    def _marker_observation(self, capture: np.ndarray,
                            initial: tuple[float, float, float]) -> tuple[np.ndarray | None, int]:
        """Détecte les marqueurs ArUco connus et en déduit une pose CAMÉRA absolue.

        --- FR : déroulé de la méthode ---
        1. Détection "grossière" directement dans l'image brute (résolution
           caméra), ce qui est nettement moins coûteux que de tout rectifier
           d'abord en vue du dessus (commentaire d'origine : on évite ainsi
           un cadre rectifié 2561x1380 et le trafic mémoire associé, coûteux
           sur un Raspberry Pi).
        2. Pour chaque tag reconnu (dont l'identifiant est dans TAG_FIELD_MM),
           on projette ses coins au sol via l'homographie caméra->sol, ce qui
           donne une position approximative en mm.
        3. Autour de cette position, on rectifie seulement une petite fenêtre
           locale de 320x320 mm à 1 mm/px, et on y refait une détection ArUco
           pour obtenir des coins beaucoup plus précis (la détection top-down
           est plus fiable qu'une détection en perspective) — sans payer le
           coût de rectifier toute l'image.
        4. On vérifie la cohérence géométrique du tag (les 4 côtés doivent
           faire environ 100 mm ; on rejette si l'erreur dépasse 30 %).
        5. On calcule la pose du robot compatible avec la position mesurée du
           tag et sa position connue sur le terrain (rotation + translation).
        6. On ne garde que les candidats proches de la pose initiale fournie
           (bornes max_shift_mm / max_yaw_deg), puis on choisit le meilleur
           selon un score combinant erreur de taille, distance et angle —
           pour qu'un tag mal identifié ne fasse pas dévier fortement le
           résultat (commentaire d'origine).

        Renvoie (pose_camera_estimée ou None, nombre_de_candidats_valides).
        """
        if self.detector is None:
            return None, 0
        height, width = capture.shape
        h = capture_homography(width, height)
        # Detect at camera resolution, then project four corners. This avoids a
        # 2561x1380 rectified frame and its extra memory traffic on the Pi.
        quads, ids, _ = self.detector.detectMarkers(capture)
        if ids is None:
            return None, 0
        candidates = []
        for quad, id_value in zip(quads, ids.ravel()):
            tag = int(id_value)
            if tag not in TAG_FIELD_MM:
                continue  # tag inconnu du terrain (bruit, autre robot, etc.).
            # Projette les 4 coins détectés (repère image) sur le plan du sol
            # (repère local droite/avant, en mm) via l'homographie caméra.
            corners = cv.perspectiveTransform(quad.astype(np.float64), h).reshape(4, 2)
            # Refine only a 320x320 mm neighborhood at 1 mm/px. This retains
            # the accuracy of top-down detection without warping the whole view.
            centre = corners.mean(axis=0)
            roi_x = math.floor(centre[0] - 160)
            roi_z = math.floor(centre[1] - 160)
            # `shift` translate simplement l'origine du sol vers le coin de la
            # petite fenêtre locale, avant de la combiner à l'homographie.
            shift = np.array([[1, 0, -roi_x], [0, 1, -roi_z], [0, 0, 1]])
            local = cv.warpPerspective(capture, shift @ h, (320, 320),
                                       flags=cv.INTER_LINEAR)
            refined, refined_ids, _ = self.detector.detectMarkers(local)
            if refined_ids is not None:
                for candidate_quad, candidate_id in zip(refined, refined_ids.ravel()):
                    if int(candidate_id) == tag:
                        # Coins affinés (détection top-down), repositionnés
                        # dans le repère sol global en ajoutant le décalage
                        # de la fenêtre locale.
                        corners = candidate_quad.reshape(4, 2).astype(np.float64)
                        corners += [roi_x, roi_z]
                        break
            # Vérifie que le tag mesure bien ~100 mm de côté (RMS des écarts
            # des 4 côtés par rapport à 100 mm, normalisé). Rejette si trop
            # déformé (mauvaise détection ou perspective extrême).
            sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            size_error = float(np.sqrt(np.mean((sides - 100.0) ** 2))) / 100.0
            if size_error > 0.30:
                continue
            # The rectified axes are opposite to robot forward/left.
            # --- FR : passage du repère "sol rectifié" (X=droite, Z=avant)
            # au repère "robot" (convention différente, d'où les signes/permutation).
            robot_corners = np.column_stack((-corners[:, 1], -corners[:, 0]))
            measured = robot_corners.mean(axis=0)  # position mesurée du centre du tag, vue du robot.
            edge = robot_corners[3] - robot_corners[0]  # un côté du tag, pour en déduire l'orientation.
            yaw = -math.degrees(math.atan2(edge[1], edge[0]))
            # On construit la rotation correspondant à ce yaw, pour convertir
            # la position "mesurée dans le repère robot" en un décalage
            # exprimé dans le repère terrain (fixe).
            angle = math.radians(yaw)
            co, si = math.cos(angle), math.sin(angle)
            offset = np.array([co * measured[0] - si * measured[1],
                               si * measured[0] + co * measured[1]])
            # Pose du robot = position connue du tag sur le terrain, moins le
            # décalage (tourné) mesuré entre le robot et ce tag.
            field_xy = np.array(TAG_FIELD_MM[tag])
            candidate = np.array([*(field_xy - offset), yaw], np.float64)
            displacement = math.hypot(*(candidate[:2] - initial[:2]))
            angular = abs(wrap_deg(candidate[2] - initial[2]))
            # On ignore un candidat trop éloigné de la pose initiale (probable
            # fausse détection, ou tag ambigu).
            if displacement <= self.max_shift_mm and angular <= self.max_yaw_deg:
                rank = 5.0 * size_error + displacement / self.max_shift_mm + angular / self.max_yaw_deg
                candidates.append((rank, candidate))
        if not candidates:
            return None, 0
        # A single incorrect tag must not pull the pose far from the prior.
        best = min(candidates, key=lambda pair: pair[0])[1]
        return best, len(candidates)

    def _rectify(self, capture: np.ndarray,
                 mm_per_px: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """Redresse (vue du dessus) la capture caméra à une résolution donnée.

        --- FR ---
        Cette méthode transforme la photo "vue en perspective" en une image
        "vue du dessus" à `mm_per_px` millimètres par pixel, avec mise en
        cache de la partie de la transformation qui ne dépend PAS du contenu
        de l'image (seulement de sa taille et de la résolution demandée) :
        homographie, empreinte au sol, masque de validité, recadrage.

        Étapes (uniquement lors du premier appel avec une clé de cache donnée) :
          1. Calcule l'homographie caméra->sol et son empreinte (ground_bounds).
          2. En déduit la taille de l'image de sortie en pixels, à `mm_per_px`.
          3. Construit la transformation globale (mise à l'échelle mm->px,
             translation pour ramener l'origine, puis l'homographie).
          4. Masque le haut de l'image source (`source_top_fraction`) : cette
             zone correspond typiquement à l'horizon / hors terrain et ne doit
             pas contribuer au recalage.
          5. Warp ce masque pour savoir quelle zone de l'image de sortie est
             réellement couverte par un pixel source valide.
          6. Recadre la sortie sur son rectangle englobant utile (`cv.boundingRect`)
             pour ne plus traiter par la suite de larges zones vides — tout en
             conservant `centre_uv`, la position du "centre" dans ce nouveau
             repère recadré, afin que les poses restent cohérentes après coup.
        """
        height, width = capture.shape
        key = (height, width, mm_per_px, self.source_top_fraction)
        if key not in self._warp_cache:
            homography = capture_homography(width, height)
            x0, z0, x1, z1 = ground_bounds(homography, (width, height))
            shape = (max(2, round((x1 - x0) / mm_per_px)),
                     max(2, round((z1 - z0) / mm_per_px)))
            # Transformation complète : (pixel image) -> (pixel sortie, à
            # mm_per_px), en composant la mise à l'échelle/translation avec
            # l'homographie caméra->sol.
            transform = np.array([[1 / mm_per_px, 0, -x0 / mm_per_px],
                                  [0, 1 / mm_per_px, -z0 / mm_per_px],
                                  [0, 0, 1]], dtype=np.float64) @ homography
            # Masque source : blanc partout, sauf la bande du haut de l'image
            # (ciel/hors-terrain), mise à zéro selon `source_top_fraction`.
            source_mask = np.full(capture.shape, 255, dtype=np.uint8)
            source_mask[:round(height * self.source_top_fraction), :] = 0
            coverage = cv.warpPerspective(source_mask, transform, shape,
                                          flags=cv.INTER_LINEAR, borderValue=0)
            patch_mask = valid_region_mask(coverage)
            rx, ry, rw, rh = cv.boundingRect(patch_mask)
            if rw < 2 or rh < 2:
                raise ValueError("Rectified capture has no valid ground-plane pixels")
            # Crop the empty perspective halo before every warp/evaluation.
            # Keep the original patch centre as a local offset so the pose
            # transform is mathematically unchanged after cropping.
            crop = np.array([[1, 0, -rx], [0, 1, -ry], [0, 0, 1]])
            centre_uv = (shape[0] / 2.0 - rx, shape[1] / 2.0 - ry)
            shape = (rw, rh)
            patch_mask = patch_mask[ry:ry + rh, rx:rx + rw].copy()
            self._warp_cache[key] = (crop @ transform, shape, patch_mask, centre_uv)
        # Récupère (ou vient de créer) la transformation mise en cache, et
        # l'applique cette fois à l'image réelle (contenu qui, lui, change
        # à chaque appel — d'où sa sortie du bloc de cache ci-dessus).
        transform, shape, patch_mask, centre_uv = self._warp_cache[key]
        patch = cv.warpPerspective(capture, transform, shape,
                                   flags=cv.INTER_LINEAR, borderValue=0)
        return patch, patch_mask, centre_uv

    def _levels_for(self, capture: np.ndarray) -> list[Level]:
        """Construit la liste des `Level` (un par étage de la pyramide) pour
        une capture donnée : rectifie la capture à chaque résolution requise
        et précalcule les quantités utiles à l'évaluation du score (carré du
        patch pour la variance, nombre de pixels valides, rayon en mm, etc.).
        """
        result = []
        for reference_float, field_mask, mm_per_px in self.reference_levels:
            patch_u8, patch_mask, centre_uv = self._rectify(capture, mm_per_px)
            patch = patch_u8.astype(np.float32)
            # Rayon (demi-diagonale) du patch en mm : sert plus tard à
            # convertir un pas de rotation (en degrés) en un déplacement de
            # pixel équivalent, pour que le gradient soit à une échelle
            # cohérente entre translation et rotation (voir `descend`).
            radius_mm = 0.5 * math.hypot(*patch.shape[::-1]) * mm_per_px
            result.append(Level(reference_float, field_mask, mm_per_px,
                                patch, cv.multiply(patch, patch), patch_mask,
                                cv.countNonZero(patch_mask), radius_mm, centre_uv))
        return result

    @staticmethod
    def _matrix(level: Level, pose_: np.ndarray) -> np.ndarray:
        """Calcule la matrice de similitude 2D (rotation + translation, pas
        d'échelle) qui, appliquée avec WARP_INVERSE_MAP, va chercher dans
        l'image de RÉFÉRENCE l'échantillon correspondant au PATCH pour la
        pose candidate donnée (x, y, yaw exprimés dans le repère "centre").

        --- FR : détail des calculs ---
        - `theta` convertit le yaw (convention "0° = axe +x, sens trigo")
          vers l'angle de rotation utilisé dans la matrice image (où l'axe v
          pointe vers le bas) : d'où le décalage de 90° et le signe.
        - `cx`, `cy` sont les coordonnées, en pixels de la référence, de la
          pose (x, y) demandée (avec origine au centre du terrain, converti
          en repère image où l'origine est en haut à gauche).
        - `tx`, `ty` sont ajustés pour que le pixel `centre_uv` du patch (son
          "origine" locale après recadrage) corresponde bien au point (cx, cy)
          de la référence, une fois la rotation appliquée.
        - Comme le warp utilisera WARP_INVERSE_MAP, cette matrice décrit la
          transformation destination (patch) -> source (référence), ce qui
          est justement ce qu'il faut pour "aller lire" l'échantillon
          correspondant dans la référence pour chaque pixel du patch.
        """
        x, y, yaw = pose_
        theta = math.radians(90.0 - yaw)
        co, si = math.cos(theta), math.sin(theta)
        cx = (x + FIELD_WIDTH_MM / 2.0) / level.mm_per_px
        cy = (FIELD_HEIGHT_MM / 2.0 - y) / level.mm_per_px
        centre_u, centre_v = level.centre_uv
        tx = cx - co * centre_u + si * centre_v
        ty = cy - si * centre_u - co * centre_v
        # WARP_INVERSE_MAP makes this destination-patch -> source-map matrix.
        return np.array([[co, -si, tx], [si, co, ty]], np.float32)

    @staticmethod
    def _evaluate(level: Level, pose_: np.ndarray,
                  sample: np.ndarray, support: np.ndarray,
                  mask: np.ndarray, square: np.ndarray,
                  cross: np.ndarray) -> tuple[float, float, float, float]:
        """Calcule le coût photométrique d'une pose candidate à un niveau donné.

        Les tableaux `sample`, `support`, `mask`, `square`, `cross` sont des
        buffers préalloués (voir `workspace` dans `locate`) réutilisés à
        chaque appel via le paramètre `dst=` d'OpenCV, pour éviter de
        réallouer de la mémoire à chaque évaluation (optimisation importante
        vu le grand nombre d'évaluations effectuées pendant l'optimisation).

        --- FR : déroulé ---
        1. Calcule la matrice de similitude pour cette pose (`_matrix`).
        2. "Tire" (warp) l'échantillon correspondant de la référence et de
           son masque de terrain dans l'espace du patch (taille identique au
           patch capturé), via WARP_INVERSE_MAP.
        3. Combine ce masque avec le masque de validité du patch lui-même
           (`bitwise_and`) : seuls les pixels valides des DEUX côtés comptent.
        4. Si trop peu de pixels valides (`count` en dessous d'un seuil), la
           pose est jugée non évaluable : coût infini.
        5. Sinon, calcule la CORRÉLATION CROISÉE NORMALISÉE (NCC) entre le
           patch et l'échantillon de référence, en ne considérant que les
           pixels du masque :
             - moyennes (mean_p, mean_r)
             - variances (var_p, var_r), via E[X²] - E[X]²
             - covariance (cov), via E[X·Y] - E[X]·E[Y]
             - corrélation = cov / sqrt(var_p * var_r), bornée à [-1, 1]
           La NCC a l'avantage d'être invariante à un gain et un offset
           globaux de luminosité (utile car l'éclairage réel de la caméra
           diffère de celui de l'image de référence).
        6. Le coût final combine :
             - la perte photométrique (1 - corrélation)
             - un terme de pénalité proportionnel à (1 - fraction de
               recouvrement), pour empêcher une pose de "gagner" en glissant
               hors du terrain physique là où il y a peu de pixels à comparer
               (commentaire d'origine).

        Renvoie (objectif, corrélation, fraction_de_recouvrement, perte_photo).
        """
        matrix = Localizer._matrix(level, pose_)
        size = level.patch.shape[::-1]
        cv.warpAffine(level.reference, matrix, size, dst=sample,
                      flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                      borderMode=cv.BORDER_CONSTANT, borderValue=0)
        cv.warpAffine(level.field_mask, matrix, size, dst=support,
                      flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
                      borderMode=cv.BORDER_CONSTANT, borderValue=0)
        cv.bitwise_and(level.patch_mask, support, dst=mask)
        count = cv.countNonZero(mask)
        fraction = count / max(level.valid_count, 1)
        if count < max(64, int(0.10 * level.valid_count)):
            return (float("inf"), 0.0, fraction, float("inf"))
        cv.multiply(sample, sample, dst=square)
        cv.multiply(level.patch, sample, dst=cross)
        mean_p = cv.mean(level.patch, mask)[0]
        mean_r = cv.mean(sample, mask)[0]
        var_p = cv.mean(level.patch_squared, mask)[0] - mean_p * mean_p
        var_r = cv.mean(square, mask)[0] - mean_r * mean_r
        if var_p < 1.0 or var_r < 1.0:
            # Zones trop uniformes (peu de texture) : la corrélation ne serait
            # pas fiable (division par une variance quasi nulle).
            return (float("inf"), 0.0, fraction, float("inf"))
        cov = cv.mean(cross, mask)[0] - mean_p * mean_r
        correlation = max(-1.0, min(1.0, cov / math.sqrt(var_p * var_r)))
        # NCC removes a global gain and offset. The overlap term prevents a
        # candidate from winning by sliding mostly off the physical field.
        photo_loss = 1.0 - correlation
        objective = photo_loss + 0.15 * (1.0 - fraction)
        return objective, correlation, fraction, photo_loss

    def locate(self, capture: np.ndarray, initial_camera_pose: tuple[float, float, float],
               prior_xy_sigma_mm: float | None = None,
               prior_yaw_sigma_deg: float | None = None) -> Result:
        """Méthode principale : estime la pose du robot à partir d'une capture.

        --- FR : vue d'ensemble de l'algorithme ---
        1. Validations d'entrée (image 8 bits niveaux de gris, pose initiale finie).
        2. Détection optionnelle de marqueurs ArUco (`_marker_observation`).
        3. Préparation des niveaux de la pyramide pour CETTE capture (`_levels_for`).
        4. Définition de deux fonctions internes :
             - `rank(candidate, score)` : score utilisé pour COMPARER/TRIER les
               candidats. En l'absence de marqueur, ajoute une pénalité de type
               "prior gaussien" qui favorise les poses proches de l'estimation
               initiale (utile pour désambiguïser un terrain qui a des
               symétries ou motifs répétés).
             - `workspace(level)` : crée les buffers réutilisables pour un
               niveau donné et renvoie une fonction `evaluate(candidate)` qui
               calcule le coût total (photométrique, et fusionné avec le
               marqueur si disponible), en rejetant les candidats hors bornes
               (`max_shift_mm`, `max_yaw_deg`) par rapport à la pose initiale.
        5. `descend(level, evaluate, pose, score)` : une descente de type
           "coordinate/gradient descent" faite à la main :
             - Calcule un gradient numérique par différences finies sur
               chacun des 3 axes (x, y, yaw), avec un pas égal à 1 pixel
               (ou son équivalent angulaire) à ce niveau de résolution.
             - Normalise ce gradient pour obtenir une direction de descente.
             - Fait une recherche de pas (line search) parmi plusieurs
               échelles décroissantes (4, 2, 1, 0.5, 0.25 pixels), et
               accepte le premier essai qui améliore strictement le score.
             - S'arrête après `self.iterations` itérations, ou plus tôt si
               le gradient est quasi nul ou qu'aucun pas n'améliore le score.
        6. Initialisation "multistart" : en l'absence de marqueur, génère une
           grille 5x5x5 de poses candidates autour de la pose initiale (au
           niveau le plus grossier), ne garde que celles avec une fraction de
           recouvrement suffisante, les trie par `rank`, puis en sélectionne
           un sous-ensemble suffisamment DIVERSIFIÉ (au moins 40 mm ou 3° les
           unes des autres) pour éviter de converger plusieurs fois vers le
           même optimum local.
        7. Boucle "coarse-to-fine" : pour chaque niveau (du plus grossier au
           plus fin), on relance `descend` depuis chaque graine, on trie les
           résultats par `rank`, et on ne garde qu'un nombre décroissant de
           graines (`keep`) au fur et à mesure qu'on affine — jusqu'à ne plus
           garder qu'une seule pose au dernier niveau.
        8. Construction du `Result` final :
             - Si un marqueur fiable est disponible : la pose est celle du
               marqueur (recalée sur l'optimum photométrique si les deux
               sont cohérents), statut "marker_fused".
             - Sinon, si la corrélation ou la fraction de recouvrement est
               trop faible, on considère l'estimation peu fiable et on
               RETOURNE LA POSE INITIALE (statut "low_confidence") plutôt
               qu'un résultat probablement erroné.
             - Sinon, on retourne la pose optimisée (statut "ok").
        """
        if capture.ndim != 2 or capture.dtype != np.uint8:
            raise ValueError("Expected an 8-bit grayscale capture")
        if not all(math.isfinite(float(v)) for v in initial_camera_pose):
            raise ValueError("Initial pose must be finite")
        sigma_xy = self.prior_xy_sigma_mm if prior_xy_sigma_mm is None else prior_xy_sigma_mm
        sigma_yaw = self.prior_yaw_sigma_deg if prior_yaw_sigma_deg is None else prior_yaw_sigma_deg
        if sigma_xy <= 0 or sigma_yaw <= 0:
            raise ValueError("Prior sigmas must be positive")
        start = perf_counter()
        # Toute l'optimisation se fait dans le repère du point "centre" (voir
        # CAMERA_PATCH_CENTRE_MM) : on convertit donc d'abord la pose initiale.
        initial_centre = np.array(camera_to_centre(initial_camera_pose), np.float64)
        marker_camera, marker_count = self._marker_observation(capture, initial_camera_pose)
        marker_centre = (None if marker_camera is None else
                         np.array(camera_to_centre(tuple(marker_camera)), np.float64))
        levels = self._levels_for(capture)
        evaluations = 0

        def rank(candidate: np.ndarray, score: tuple[float, float, float, float]) -> float:
            """Score de tri/sélection des candidats (pas forcément identique
            à l'objectif optimisé par `descend`, qui lui utilise `score[0]`).
            Sans marqueur, on ajoute un terme de prior gaussien qui pénalise
            l'éloignement (position ET angle) par rapport à la pose initiale,
            normalisé par les écarts-types `sigma_xy` / `sigma_yaw`.
            """
            if marker_centre is not None:
                return score[0]
            camera = centre_to_camera(tuple(candidate))
            distance = math.hypot(camera[0] - initial_camera_pose[0],
                                  camera[1] - initial_camera_pose[1])
            angle = abs(wrap_deg(camera[2] - initial_camera_pose[2]))
            return (score[0] + 0.03 * (distance / sigma_xy) ** 2
                    + 0.03 * (angle / sigma_yaw) ** 2)

        def workspace(level: Level):
            """Crée les buffers réutilisables pour un niveau, et renvoie une
            fonction `evaluate` fermée sur ces buffers (évite les réallocations
            mémoire répétées d'un appel à l'autre — voir _evaluate)."""
            shape = level.patch.shape
            sample = np.empty(shape, np.float32)
            support = np.empty(shape, np.uint8)
            mask = np.empty(shape, np.uint8)
            square = np.empty(shape, np.float32)
            cross = np.empty(shape, np.float32)

            def evaluate(candidate: np.ndarray) -> tuple[float, float, float, float]:
                nonlocal evaluations
                evaluations += 1
                # Rejette immédiatement (coût infini) tout candidat qui, une
                # fois reconverti en pose caméra, sortirait des bornes de
                # recherche autorisées (max_shift_mm / max_yaw_deg) par
                # rapport à la pose initiale — évite d'explorer des poses
                # physiquement improbables.
                camera = centre_to_camera(tuple(candidate))
                if (math.hypot(camera[0] - initial_camera_pose[0],
                               camera[1] - initial_camera_pose[1]) > self.max_shift_mm or
                    abs(wrap_deg(camera[2] - initial_camera_pose[2])) > self.max_yaw_deg):
                    return float("inf"), 0.0, 0.0, float("inf")
                registration_loss, corr, fraction, photo_loss = self._evaluate(
                    level, candidate, sample, support, mask, square, cross)
                if marker_centre is None:
                    return registration_loss, corr, fraction, photo_loss
                # --- FR : fusion avec le marqueur ---
                # Quand un marqueur ArUco fiable est disponible, l'objectif
                # optimisé devient principalement l'écart (position + angle)
                # à la pose donnée par le marqueur (`marker_loss`), avec
                # seulement 5 % de poids pour la perte photométrique — qui
                # sert alors de régularisation face à des occultants 3D
                # inévitables (objets qui dépassent du plan du sol et
                # perturbent la corrélation photométrique pure).
                delta_xy = candidate[:2] - marker_centre[:2]
                angle = wrap_deg(candidate[2] - marker_centre[2])
                marker_loss = float(delta_xy @ delta_xy) / 2500.0 + (angle / 5.0) ** 2
                # The tag regularizes a photo loss with unavoidable 3-D occluders.
                return (marker_loss + 0.05 * (registration_loss if math.isfinite(registration_loss) else 1.0),
                        corr, fraction, photo_loss)

            return evaluate

        def descend(level: Level, evaluate, pose_: np.ndarray,
                    score: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
            """Descente locale par gradient numérique + recherche de pas.

            --- FR ---
            `units` définit, pour chaque axe (x, y, yaw), le "pas de base" :
            1 pixel en mm pour x/y, et l'équivalent angulaire d'1 pixel pour
            le yaw (en utilisant le rayon du patch : un pixel de décalage au
            bord du patch correspond à peu près à `mm_per_px / rayon` radians
            de rotation — d'où le `yaw_per_px` calculé via une conversion
            radian -> degré `* 180/pi`... note : ici c'est l'inverse, le pas
            est en degrés directement).

            À chaque itération :
              - Pour chaque axe, on évalue le coût en pose+delta et pose-delta
                (différence finie centrée) pour estimer la dérivée partielle.
                Si un seul côté est évaluable (l'autre étant hors bornes ou
                sans texture), on utilise une différence "avant" ou "arrière"
                à la place (moins précise mais permet de continuer à progresser
                près des bords du domaine de recherche).
              - On normalise le gradient pour obtenir une direction unitaire
                de plus forte descente.
              - On teste plusieurs longueurs de pas décroissantes (recherche
                de pas façon "backtracking line search") et on accepte le
                premier essai qui améliore strictement le score (à 1e-6 près).
              - Si aucune amélioration n'est trouvée, ou si le gradient est
                quasi nul, on arrête (convergence locale atteinte).
            """
            yaw_per_px = level.mm_per_px / max(level.radius_mm * math.pi / 180.0, 1.0)
            units = np.array([level.mm_per_px, level.mm_per_px, yaw_per_px])
            for _ in range(self.iterations):
                gradient = np.zeros(3, np.float64)
                for axis in range(3):
                    delta = np.zeros(3)
                    delta[axis] = units[axis]
                    plus = evaluate(pose_ + delta)[0]
                    minus = evaluate(pose_ - delta)[0]
                    if math.isfinite(plus) and math.isfinite(minus):
                        gradient[axis] = (plus - minus) / 2.0
                    elif math.isfinite(plus):
                        gradient[axis] = plus - score[0]
                    elif math.isfinite(minus):
                        gradient[axis] = score[0] - minus
                magnitude = float(np.linalg.norm(gradient))
                if magnitude < 1e-6:
                    break
                direction = -gradient / magnitude
                improved = False
                for step in (4.0, 2.0, 1.0, 0.5, 0.25):
                    trial = pose_ + direction * units * step
                    trial_score = evaluate(trial)
                    if trial_score[0] < score[0] - 1e-6:
                        pose_, score = trial, trial_score
                        improved = True
                        break
                if not improved:
                    break
            return pose_, score

        # --- FR : point de départ + graines "multistart" -------------------
        # On évalue toujours la pose initiale elle-même en premier.
        first_evaluate = workspace(levels[0])
        seeds: list[tuple[np.ndarray, tuple[float, float, float, float]]] = [
            (initial_centre, first_evaluate(initial_centre))]
        if marker_centre is None and self.multistart > 1:
            # Sans marqueur, le recalage photométrique seul peut se piéger
            # dans un optimum local (motifs répétés du terrain, symétries...).
            # On construit donc une grille de points de départ alternatifs
            # autour de la pose initiale, à évaluer au niveau le plus grossier
            # (le moins coûteux) avant de les affiner.
            xy_span = min(self.max_shift_mm, max(150.0, 1.5 * sigma_xy))
            yaw_span = min(self.max_yaw_deg, max(10.0, 1.5 * sigma_yaw))
            proposals = []
            for dx in (-xy_span, -xy_span / 2, 0.0, xy_span / 2, xy_span):
                for dy in (-xy_span, -xy_span / 2, 0.0, xy_span / 2, xy_span):
                    for da in (-yaw_span, -yaw_span / 2, 0.0,
                               yaw_span / 2, yaw_span):
                        camera = (initial_camera_pose[0] + dx,
                                  initial_camera_pose[1] + dy,
                                  initial_camera_pose[2] + da)
                        centre = np.array(camera_to_centre(camera), np.float64)
                        trial_score = first_evaluate(centre)
                        # On ne garde que les candidats évaluables et avec un
                        # minimum de recouvrement (20 %) pour être crédibles.
                        if math.isfinite(trial_score[0]) and trial_score[2] >= 0.20:
                            proposals.append((rank(centre, trial_score), centre, trial_score))
            proposals.sort(key=lambda item: item[0])
            for _, centre, trial_score in proposals:
                # N'ajoute une graine que si elle est suffisamment différente
                # de toutes celles déjà retenues (>40 mm ou >3°), pour éviter
                # de "gaspiller" du calcul sur plusieurs graines qui vont
                # converger vers le même optimum local.
                if all(math.hypot(*(centre[:2] - other[:2])) > 40.0 or
                       abs(wrap_deg(centre[2] - other[2])) > 3.0
                       for other, _ in seeds):
                    seeds.append((centre, trial_score))
                if len(seeds) >= self.multistart:
                    break

        # --- FR : optimisation "coarse-to-fine" -----------------------------
        # On parcourt les niveaux du plus grossier au plus fin (`levels` a été
        # construit dans cet ordre). À chaque niveau, on relance une descente
        # locale depuis chacune des graines courantes, on trie les résultats,
        # puis on réduit progressivement le nombre de graines conservées
        # (`keep`) à mesure qu'on approche du niveau le plus fin — sauf si un
        # marqueur est disponible, auquel cas une seule graine suffit dès le
        # départ (la recherche est bien moins sujette aux optima locaux).
        for index, level in enumerate(levels):
            evaluate = first_evaluate if index == 0 else workspace(level)
            refined = []
            for pose_, old_score in seeds:
                # Au niveau 0, le score de la graine est déjà connu (calculé
                # plus haut) ; aux niveaux suivants, il faut le recalculer car
                # la fonction de coût change de résolution.
                score = old_score if index == 0 else evaluate(pose_)
                if math.isfinite(score[0]):
                    refined.append(descend(level, evaluate, pose_, score))
            if not refined:
                # Aucune graine n'a pu être évaluée à ce niveau : abandon,
                # avec un statut explicite plutôt qu'un résultat inventé.
                return Result(*initial_camera_pose, float("inf"), 0.0, 0.0,
                              (perf_counter() - start) * 1000, evaluations,
                              "insufficient_overlap_or_texture", marker_count,
                              float("inf"))
            refined.sort(key=lambda item: rank(item[0], item[1]))
            if marker_centre is not None:
                keep = 1
            elif index + 1 == len(levels):
                keep = 1
            elif index + 2 == len(levels):
                keep = min(3, self.multistart)
            else:
                keep = self.multistart
            seeds = refined[:keep]

        # --- FR : construction du résultat final ----------------------------
        pose_, score = seeds[0]
        if marker_camera is not None:
            # Cas "marqueur disponible" : on part de la pose photométrique
            # finale, mais on la RECALE sur la pose du marqueur si elle s'en
            # écarte trop (>40 mm ou >3°) — le marqueur, plus fiable, a alors
            # le dernier mot. On réévalue ensuite le score au niveau le plus
            # fin pour renvoyer des métriques cohérentes avec la pose choisie.
            x, y, yaw = centre_to_camera(tuple(pose_))
            if (math.hypot(x - marker_camera[0], y - marker_camera[1]) > 40 or
                abs(wrap_deg(yaw - marker_camera[2])) > 3):
                x, y, yaw = marker_camera
            final_pose = np.array(camera_to_centre((x, y, yaw)), np.float64)
            final_score = workspace(levels[-1])(final_pose)
            return Result(x, y, wrap_deg(yaw), final_score[3], final_score[1],
                          final_score[2], (perf_counter() - start) * 1000,
                          evaluations, "marker_fused", marker_count, final_score[0])
        if score[1] < 0.30 or score[2] < 0.20:
            # Corrélation ou recouvrement trop faibles : on ne fait pas
            # confiance au résultat optimisé et on retourne la pose initiale
            # inchangée, avec le statut "low_confidence" pour le signaler
            # clairement à l'appelant plutôt que de renvoyer une estimation
            # potentiellement fausse en silence.
            initial_score = workspace(levels[-1])(initial_centre)
            return Result(*initial_camera_pose, initial_score[3], initial_score[1],
                          initial_score[2], (perf_counter() - start) * 1000,
                          evaluations, "low_confidence", 0, initial_score[0])
        x, y, yaw = centre_to_camera(tuple(pose_))
        return Result(x, y, yaw, score[3], score[1], score[2],
                      (perf_counter() - start) * 1000, evaluations,
                      "ok", 0, score[0])


# --- Main ------------------------------------------------------------------


def build_parser(parser) -> None:
    parser.add_argument("--reference", type=Path, default=None,
                        help="Reference field map. Default: input/FieldBW.png.")
    parser.add_argument("--max-reference-side", type=int, default=384,
                        help="Longest side of the finest pyramid level.")
    parser.add_argument("--levels", type=int, default=3,
                        help="Pyramid depth, coarse to fine.")
    parser.add_argument("--iterations", type=int, default=14,
                        help="Gradient-descent iterations per level.")
    parser.add_argument("--source-top-fraction", type=float, default=0.15,
                        help="Top band of the capture to mask out (sky/off-field).")
    parser.add_argument("--multistart", type=int, default=5)
    parser.add_argument("--prior-xy-sigma-mm", type=float, default=100.0)
    parser.add_argument("--prior-yaw-sigma-deg", type=float, default=5.0)
    parser.add_argument("--max-shift-mm", type=float, default=600.0)
    parser.add_argument("--max-yaw-deg", type=float, default=45.0)
    parser.add_argument("--opencv-threads", type=int, default=1,
                        help="OpenCV workers; one avoids worker overhead for small Pi patches")
    parser.add_argument("--initial-dx", type=float, default=0)
    parser.add_argument("--initial-dy", type=float, default=0)
    parser.add_argument("--initial-dyaw", type=float, default=0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--no-markers", action="store_true",
                        help="Skip ArUco fusion and solve photometrically only.")


def run(captures: list[str], out_dir: str, args) -> int:
    """Traite un lot de captures dont le nom de fichier encode la vérité
    terrain (x, y, yaw), calcule la pose estimée pour chacune, affiche les
    erreurs, et écrit optionnellement un CSV récapitulatif (utile pour
    évaluer les performances de l'algorithme sur un jeu de captures de test).
    """
    if args.opencv_threads < 1:
        print("[error] --opencv-threads must be positive", file=sys.stderr)
        return 2
    # Un seul thread OpenCV : pour de petits patchs (Raspberry Pi), le coût de
    # synchronisation entre threads dépasserait le gain de parallélisation
    # (voir l'aide de --opencv-threads).
    cv.setNumThreads(args.opencv_threads)

    reference = args.reference or Path(paths.map_path("gradient"))
    if not reference.exists():
        print(f"[error] reference map not found: {reference}", file=sys.stderr)
        return 2

    localizer = Localizer(read_gray(reference), args.max_reference_side,
                          args.levels, args.iterations,
                          max_shift_mm=args.max_shift_mm,
                          max_yaw_deg=args.max_yaw_deg,
                          use_markers=not args.no_markers,
                          source_top_fraction=args.source_top_fraction,
                          multistart=args.multistart,
                          prior_xy_sigma_mm=args.prior_xy_sigma_mm,
                          prior_yaw_sigma_deg=args.prior_yaw_sigma_deg)
    rows = []
    for path in captures:
        try:
            # `truth` = vérité terrain lue dans le nom du fichier de capture.
            truth = parse_pose(Path(path))
            # On simule une estimation initiale imparfaite en perturbant la
            # vérité terrain avec --initial-dx/dy/dyaw (utile pour tester la
            # robustesse de l'algorithme face à une mauvaise pose de départ).
            initial = (truth[0] + args.initial_dx, truth[1] + args.initial_dy,
                       truth[2] + args.initial_dyaw)
            result = localizer.locate(read_gray(Path(path)), initial)
            row = {"capture": os.path.basename(path), "status": result.status,
                   "x_mm": result.x_mm, "y_mm": result.y_mm,
                   "yaw_deg": result.yaw_deg, "loss": result.loss,
                   "objective": result.objective,
                   "correlation": result.correlation,
                   "valid_fraction": result.valid_fraction,
                   "elapsed_ms": result.elapsed_ms,
                   "evaluations": result.evaluations,
                   "marker_count": result.marker_count,
                   "error_mm": math.hypot(result.x_mm - truth[0],
                                          result.y_mm - truth[1]),
                   "yaw_error_deg": abs(wrap_deg(result.yaw_deg - truth[2]))}
            rows.append(row)
            if not args.quiet:
                print(f"{os.path.basename(path)}: {result.status} "
                      f"x={result.x_mm:.1f} y={result.y_mm:.1f} "
                      f"yaw={result.yaw_deg:.1f} error={row['error_mm']:.1f} mm/"
                      f"{row['yaw_error_deg']:.1f} deg "
                      f"time={result.elapsed_ms:.1f} ms")
        except (ValueError, FileNotFoundError) as exc:
            print(f"{os.path.basename(path)}: {exc}")

    if rows:
        errors = np.array([r["error_mm"] for r in rows])
        yaw_errors = np.array([r["yaw_error_deg"] for r in rows])
        times = np.array([r["elapsed_ms"] for r in rows])
        print(f"\nLocalized {len(rows)}/{len(captures)} capture(s).")
        print(f"Timing    : {times.mean():.0f} ms mean, "
              f"{np.median(times):.0f} ms median, {times.max():.0f} ms max")
        print(f"Error     : position mean {errors.mean():.1f} / "
              f"max {errors.max():.1f} mm, heading mean {yaw_errors.mean():.2f} / "
              f"max {yaw_errors.max():.2f} deg")

    if args.csv and rows:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV       : {args.csv}")

    return 0 if rows else 1


if __name__ == "__main__":
    # Run standalone with the same flags the harness would pass along.
    import argparse

    _parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _parser.add_argument("captures", nargs="*")
    _parser.add_argument("--no-output", action="store_true")
    _parser.add_argument("--quiet", action="store_true")
    build_parser(_parser)
    _args = _parser.parse_args(sys.argv[1:])
    _paths = ([paths.resolve_capture("gradient", n) for n in _args.captures]
              or paths.captures("gradient"))
    raise SystemExit(run(_paths, paths.output_dir("gradient"), _args))
