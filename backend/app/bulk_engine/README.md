# Bulk Engine API

Bulk Engine provides batch analytics and visualization endpoints for cell data
stored in the database (table `cells`). All routes are served under the backend
API prefix: `/api/v1`.

## Base URL

```
http://<host>:<port>/api/v1
```

## Common Query Parameters

- `dbname` (required): database file name, e.g. `example.db`
- `label` (optional): filter by manual label
  - empty or `all` disables filtering
  - `N/A` matches both `N/A` and `1000`
- `channel`:
  - heatmap endpoints: `fluo1 | fluo2`
  - intensity endpoints: `ph | fluo1 | fluo2`
- `degree` (heatmap endpoints): integer >= 1
- `center_ratio` (HU separation overlay): float 0.0-1.0
- `max_to_min_ratio` (HU separation overlay): float >= 0.0

## Endpoints

### Heatmap and Visualization

**GET** `/get-heatmap-vectors-csv`  
Returns CSV bytes for heatmap vectors (u1/G pairs per cell).  
Params: `dbname`, `label`, `channel`, `degree`  
Response: `text/csv`

**GET** `/get-heatmap-abs-plot`  
Absolute heatmap plot.  
Params: `dbname`, `label`, `channel`, `degree`  
Response: `image/png`

**GET** `/get-heatmap-rel-plot`  
Relative heatmap plot.  
Params: `dbname`, `label`, `channel`, `degree`  
Response: `image/png`

**GET** `/get-hu-separation-overlay`  
HU separation overlay plot.  
Params: `dbname`, `label`, `channel`, `degree`, `center_ratio`, `max_to_min_ratio`  
Response: `image/png`

**GET** `/get-map256-strip`  
Map256 strip image across cells.  
Params: `dbname`, `label`, `channel`, `degree`  
Response: `image/png`

**GET** `/get-contours-grid-plot`  
Contours grid plot aligned to the cell principal axes.  
Params: `dbname`, `label`  
Response: `image/png`

**GET** `/get-contours-grid-csv`  
Contours grid coordinates (u1/u2 per point).  
Params: `dbname`, `label`  
Response: `text/csv`

### Measurements

**GET** `/get-cell-lengths`  
Cell lengths (um) using PCA over contour.  
Params: `dbname`, `label`  
Response: JSON array of `{ "cell_id": string, "length": number }`

**GET** `/get-cell-lengths-plot`  
Boxplot for cell lengths.  
Params: `dbname`, `label`  
Response: `image/png`

**GET** `/get-cell-areas`  
Cell areas (px^2).  
Params: `dbname`, `label`  
Response: JSON array of `{ "cell_id": string, "area": number }`

**GET** `/get-cell-areas-plot`  
Boxplot for cell areas.  
Params: `dbname`, `label`  
Response: `image/png`

**GET** `/get-normalized-medians`  
Normalized median intensities per cell.  
Params: `dbname`, `label`, `channel`  
Response: JSON array of `{ "cell_id": string, "normalized_median": number }`

**GET** `/get-normalized-medians-plot`  
Boxplot for normalized medians.  
Params: `dbname`, `label`, `channel`  
Response: `image/png`

**GET** `/get-raw-intensities`  
Raw intensity values inside each contour.  
Params: `dbname`, `label`, `channel`  
Response: JSON array of `{ "cell_id": string, "intensities": number[] }`

## Example Requests

Heatmap vectors CSV:

```sh
curl -G \
  "http://localhost:3000/api/v1/get-heatmap-vectors-csv" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "label=all" \
  --data-urlencode "channel=fluo1" \
  --data-urlencode "degree=4" \
  -o heatmap.csv
```

Cell lengths (JSON):

```sh
curl -G \
  "http://localhost:3000/api/v1/get-cell-lengths" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "label=1"
```

Heatmap plot (PNG):

```sh
curl -G \
  "http://localhost:3000/api/v1/get-heatmap-abs-plot" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "label=all" \
  --data-urlencode "channel=fluo1" \
  --data-urlencode "degree=4" \
  -o heatmap.png
```

## Notes

- Heatmap endpoints use a single-worker process pool to avoid concurrent heavy
  CPU work; large datasets may take time to respond.
- Some endpoints trigger Slack notifications when configured.
- Errors:
  - `404` if the database is missing or no data is available
  - `400` for invalid parameters (e.g., unsupported channel or degree)
  - `500` for unexpected failures
