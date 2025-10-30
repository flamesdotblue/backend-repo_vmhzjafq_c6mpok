import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from database import db, create_document, get_documents
from schemas import Village, Boundary, Road, FeatureCollection
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "India Mapping Backend"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or ""
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response

# -------------------- Upload Endpoints --------------------

class VillagesPayload(BaseModel):
    villages: List[Village]

@app.post("/upload/villages")
def upload_villages(payload: VillagesPayload):
    inserted = 0
    for v in payload.villages:
        create_document('village', v)
        inserted += 1
    return {"inserted": inserted}

@app.post("/upload/boundaries")
def upload_boundaries(fc: FeatureCollection):
    if fc.type != 'FeatureCollection':
        raise HTTPException(status_code=400, detail="Expected FeatureCollection")
    inserted = 0
    for f in fc.features:
        # Basic validation
        if f.get('type') != 'Feature' or 'geometry' not in f:
            continue
        create_document('boundary', f)
        inserted += 1
    return {"inserted": inserted}

@app.post("/upload/roads")
def upload_roads(fc: FeatureCollection):
    if fc.type != 'FeatureCollection':
        raise HTTPException(status_code=400, detail="Expected FeatureCollection")
    inserted = 0
    for f in fc.features:
        if f.get('type') != 'Feature' or 'geometry' not in f:
            continue
        create_document('road', f)
        inserted += 1
    return {"inserted": inserted}

# -------------------- Filters --------------------

@app.get("/filters/states")
def get_states():
    feats = get_documents('boundary', {})
    states = set()
    for f in feats:
        p = f.get('properties', {})
        s = p.get('state') or p.get('STATE') or p.get('State')
        if s:
            states.add(s)
    return sorted(states)

@app.get("/filters/districts")
def get_districts(state: Optional[str] = None):
    q = {}
    feats = get_documents('boundary', q)
    districts = set()
    for f in feats:
        p = f.get('properties', {})
        s = p.get('state') or p.get('STATE') or p.get('State')
        if state and s != state:
            continue
        d = p.get('district') or p.get('DISTRICT') or p.get('District')
        if d:
            districts.add(d)
    return sorted(districts)

@app.get("/filters/subdistricts")
def get_subdistricts(state: Optional[str] = None, district: Optional[str] = None):
    feats = get_documents('boundary', {})
    subs = set()
    for f in feats:
        p = f.get('properties', {})
        s = p.get('state') or p.get('STATE') or p.get('State')
        d = p.get('district') or p.get('DISTRICT') or p.get('District')
        sd = p.get('subdistrict') or p.get('SUBDIST') or p.get('Subdistrict') or p.get('Sub_District')
        if state and s != state:
            continue
        if district and d != district:
            continue
        if sd:
            subs.add(sd)
    return sorted(subs)

# -------------------- Data Retrieval --------------------

@app.get("/data/subdistrict")
def get_subdistrict_data(state: str, district: str, subdistrict: str):
    # Boundaries
    bounds = []
    for f in get_documents('boundary', {}):
        p = f.get('properties', {})
        s = p.get('state') or p.get('STATE') or p.get('State')
        d = p.get('district') or p.get('DISTRICT') or p.get('District')
        sd = p.get('subdistrict') or p.get('SUBDIST') or p.get('Subdistrict') or p.get('Sub_District')
        if s == state and d == district and sd == subdistrict:
            bounds.append(f)

    # Villages
    villages = get_documents('village', {
        'state': state,
        'district': district,
        'subdistrict': subdistrict
    })

    # Roads (no strict filter stored; return all for now)
    roads = get_documents('road', {})

    return {
        'boundaries': { 'type': 'FeatureCollection', 'features': bounds },
        'villages': villages,
        'roads': { 'type': 'FeatureCollection', 'features': roads }
    }

# -------------------- Analysis --------------------

class ClusterParams(BaseModel):
    metric: str = 'population'
    breaks: List[float] = [1000.0, 5000.0]
    maxDistanceKm: Optional[float] = None
    villages: List[Dict[str, Any]]

@app.post('/analyze/cluster')
def analyze_cluster(params: ClusterParams):
    b1, b2 = params.breaks[0], params.breaks[1]
    metric = params.metric
    result = []
    # compute centroid for distance if provided
    if params.villages:
        cx = sum(v.get('lon', 0) for v in params.villages) / len(params.villages)
        cy = sum(v.get('lat', 0) for v in params.villages) / len(params.villages)
    else:
        cx, cy = 0.0, 0.0

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    for v in params.villages:
        value = float(v.get(metric, 0) or 0)
        clusterIdx = 0 if value < b1 else (1 if value < b2 else 2)
        dist = haversine(v.get('lat', 0), v.get('lon', 0), cy, cx)
        within = True
        if params.maxDistanceKm is not None:
            within = dist <= params.maxDistanceKm
        result.append({ **v, 'clusterIdx': clusterIdx, 'distanceKm': dist, 'withinMaxDistance': within })

    return { 'clusters': result }

# -------------------- Road Generation --------------------

class RoadGenParams(BaseModel):
    villages: List[Dict[str, Any]]

@app.post('/generate/roads')
def generate_roads(params: RoadGenParams):
    # Build a simple Minimum Spanning Tree (Prim) over village coordinates
    pts = [(v.get('lon'), v.get('lat')) for v in params.villages if v.get('lon') is not None and v.get('lat') is not None]
    n = len(pts)
    if n == 0:
        return {'type': 'FeatureCollection', 'features': []}

    def dist(a, b):
        x1, y1 = a; x2, y2 = b
        # approximate planar distance in degrees (sufficient for small areas)
        dx = (x2 - x1) * math.cos(math.radians((y1 + y2) / 2))
        dy = (y2 - y1)
        return math.hypot(dx, dy)

    in_mst = [False]*n
    d = [float('inf')]*n
    parent = [-1]*n
    d[0] = 0
    for _ in range(n):
        u = -1
        best = float('inf')
        for i in range(n):
            if not in_mst[i] and d[i] < best:
                best = d[i]; u = i
        if u == -1:
            break
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v]:
                w = dist(pts[u], pts[v])
                if w < d[v]:
                    d[v] = w
                    parent[v] = u

    features = []
    for i in range(1, n):
        if parent[i] != -1:
            a = pts[i]
            b = pts[parent[i]]
            features.append({
                'type': 'Feature',
                'properties': { 'generated': True },
                'geometry': { 'type': 'LineString', 'coordinates': [ [a[0], a[1]], [b[0], b[1]] ] }
            })

    return { 'type': 'FeatureCollection', 'features': features }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
