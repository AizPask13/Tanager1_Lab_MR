# Tanager-1 Technical Scientific Brief

## Scope
This project uses orthorectified surface reflectance from Tanager-1 (20250501_143138_87_4001) over Mato Grosso, Brazil, to infer plant physiological gradients at lot scale and intra-lot scale.

## Scientific framing
- The analysis is physiology-informed, not laboratory-validated biochemistry.
- Absolute chlorophyll estimates are literature-linked proxies.
- Nitrogen, water, biomass, stress and efficiency variables remain scene-relative rankings unless field calibration is added.
- The strength of Tanager-1 is the combination of red-edge, NIR and SWIR, which separates pigment, function, water and structure.

## Physiological interpretation
- Visible bands capture chlorophyll and carotenoid absorption.
- The red-edge captures chlorophyll concentration, canopy nitrogen status and internal leaf structure.
- NIR is dominated by mesophyll scattering and canopy architecture.
- SWIR captures water absorption and dry-matter chemistry such as cellulose and lignin related signals.

## Current scene highlights
- Lots analysed: 66
- Strongest chlorophyll lots: A23, A56, A55, A45, A64
- Lowest chlorophyll lots: A61, A65, A62, A39, A36
- Main PCA dimensions are driven by water and structural chemistry, not only greenness.
- Lots with largest critical subzone share: A62, A16, A65, A35, A48, A42, A39, A40

## Added competition-oriented upgrades
1. Spectral uncertainty by lot from surface reflectance uncertainty bands.
2. Red-edge shape metrics and first derivative analysis.
3. Robust anomaly scoring across physiology, PCA and heterogeneity.
4. Intra-lot management subzones from NDRE, REIP and WBI.
5. Automatic interpretation and management recommendation for every lot.

## Limitations
- PLSR targets come from spectral proxies of the same scene, so that model is internal consistency, not external validation.
- No destructive leaf chemistry or field fluorometer data is available yet.
- Phenological differences can mimic stress or nutrient contrasts in some lots.
- Atmospheric or orthorectification residuals may affect SWIR stability in specific polygons.

## Why this is competitive
The workflow moves beyond generic vegetation mapping. It turns Tanager-1 spectral richness into actionable crop physiology layers:
- lot ranking
- red-edge function
- uncertainty-aware interpretation
- anomaly detection
- intra-lot management zoning

## Next validation step
Collect field measurements for chlorophyll, leaf N, relative water content and crop stage in a small subset of representative lots, especially the anomalies and the high-gradient subzones.
