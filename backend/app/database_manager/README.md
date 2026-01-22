# Database Manager API

Database Manager handles SQLite database files and cell-level queries, images,
and exports. All routes are served under the backend API prefix: `/api/v1`.

## Base URL

```
http://<host>:<port>/api/v1
```

## Database Files

- Stored in `backend/app/databases/`.
- File names must end with `.db` (path components are stripped).
- Upload triggers a lightweight migration to ensure required columns exist.

## Common Query Parameters

- `dbname` (required): database filename, e.g. `example.db`
- `cell_id` (required for cell endpoints): cell identifier in the `cells` table
- `image_type`:
  - image endpoints: `ph | fluo1 | fluo2` (some are fluo-only; see below)
- `draw_contour` (optional): `true | false`
- `draw_scale_bar` (optional): `true | false`
- `degree` (optional): integer >= 1 (default: `4`)

## Endpoints

### Database Files

**GET** `/get-databases`  
Lists available database files.  
Response: JSON array of filenames.

**POST** `/database_files`  
Upload a database file (multipart form field `file`).  
Response: `{ "filename": "example.db" }`

**GET** `/database_files/{dbname}`  
Download a database file.  
Response: `application/octet-stream`

**DELETE** `/database_files/{dbname}`  
Delete a database file.  
Response: `{ "deleted": true, "filename": "example.db" }`

### Labels and Contours

**GET** `/get-cell-ids`  
Returns all cell IDs.  
Params: `dbname`  
Response: JSON array of strings.

**GET** `/get-cell-ids-by-label`  
Returns cell IDs matching a manual label.  
Params: `dbname`, `label`  
Response: JSON array of strings.

**GET** `/get-manual-labels`  
Returns distinct manual labels.  
Params: `dbname`  
Response: JSON array of strings.

**GET** `/get-cell-contour`  
Returns contour points.  
Params: `dbname`, `cell_id`  
Response: `{ "contour": [[x, y], ...] }`

**GET** `/get-cell-label`  
Returns the manual label for a cell.  
Params: `dbname`, `cell_id`  
Response: string.

**PATCH** `/update-cell-label`  
Updates the manual label for a cell.  
Params: `dbname`, `cell_id`, `label`  
Response: `{ "cell_id": "<id>", "label": "<label>" }`

**PATCH** `/elastic-contour`  
Dilate (`delta > 0`) or erode (`delta < 0`) the stored contour.  
Params: `dbname`, `cell_id`, `delta`  
Response: `{ "cell_id": "<id>", "contour": [[x, y], ...] }`

### Images

**GET** `/get-cell-image`  
Raw cell image with optional contour/scale bar.  
Params: `dbname`, `cell_id`, `image_type`, `draw_contour`, `draw_scale_bar`  
Response: `image/png`

**GET** `/get-cell-image-optical-boost`  
Contrast-normalized fluo image (fluo-only).  
Params: `dbname`, `cell_id`, `image_type=fluo1|fluo2`, `draw_contour`, `draw_scale_bar`  
Response: `image/png`

**GET** `/get-cell-overlay`  
Ph image with fluo overlay inside the contour.  
Params: `dbname`, `cell_id`  
Response: `image/png`

### Derived Visualizations

**GET** `/get-cell-replot`  
Replot image with contour alignment.  
Params: `dbname`, `cell_id`, `image_type=ph|fluo1|fluo2`, `degree`, `dark_mode`  
Response: `image/png`

**GET** `/get-cell-heatmap`  
Heatmap image (fluo-only).  
Params: `dbname`, `cell_id`, `image_type=fluo1|fluo2`, `degree`  
Response: `image/png`

**GET** `/get-cell-map256`  
Map256 image (fluo-only).  
Params: `dbname`, `cell_id`, `image_type=fluo1|fluo2`, `degree`  
Response: `image/png`

**GET** `/get-cell-map256-jet`  
Map256 jet-colored image (fluo-only).  
Params: `dbname`, `cell_id`, `image_type=fluo1|fluo2`, `degree`  
Response: `image/png`

**GET** `/get-cell-distribution`  
Intensity distribution plot for a cell.  
Params: `dbname`, `cell_id`, `image_type=ph|fluo1|fluo2`  
Response: `image/png`

### Exports

**GET** `/get-annotation-zip`  
ZIP of contour overlays and a manifest.  
Params: `dbname`, `image_type=ph|fluo1|fluo2`, `raw`, `downscale`  
Response: `application/zip`

Notes:
- When `raw=false` and `image_type=ph`, images are downscaled (default `0.2`).
- `downscale` is ignored for fluo images or when `raw=true`.
- ZIP contains `images/<cell_id>.png` and `manifest.json`.

## Example Requests

Upload a database:

```sh
curl -X POST \
  "http://localhost:3000/api/v1/database_files" \
  -F "file=@/path/to/example.db"
```

Fetch a cell image with contour:

```sh
curl -G \
  "http://localhost:3000/api/v1/get-cell-image" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "cell_id=cell-001" \
  --data-urlencode "image_type=ph" \
  --data-urlencode "draw_contour=true" \
  -o cell.png
```

Update a label:

```sh
curl -X PATCH \
  "http://localhost:3000/api/v1/update-cell-label" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "cell_id=cell-001" \
  --data-urlencode "label=1"
```

Download annotations:

```sh
curl -G \
  "http://localhost:3000/api/v1/get-annotation-zip" \
  --data-urlencode "dbname=example.db" \
  --data-urlencode "image_type=ph" \
  -o annotations.zip
```

## Notes

- Heatmap and map256 endpoints run in a single-worker process pool; responses
  can queue under heavy load.
- Annotation ZIP generation runs in a separate process pool to avoid blocking
  the web server.
- Errors:
  - `400` invalid input (e.g., bad `dbname`, unsupported `image_type`)
  - `404` database or cell not found
  - `500` unexpected failure
