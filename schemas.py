from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Village(BaseModel):
    name: str = Field('', description="Village name")
    lat: float = Field(..., description="Latitude in WGS84")
    lon: float = Field(..., description="Longitude in WGS84")
    population: Optional[float] = Field(0, ge=0)
    area: Optional[float] = Field(0, ge=0, description="Area in sqkm or hectares as provided")
    state: Optional[str] = ''
    district: Optional[str] = ''
    subdistrict: Optional[str] = ''

class GeometryFeature(BaseModel):
    type: str
    coordinates: Any

class Boundary(BaseModel):
    properties: Dict[str, Any] = {}
    geometry: GeometryFeature
    type: str = 'Feature'

class Road(BaseModel):
    properties: Dict[str, Any] = {}
    geometry: GeometryFeature
    type: str = 'Feature'

class FeatureCollection(BaseModel):
    type: str = 'FeatureCollection'
    features: List[Dict[str, Any]]

# Note: class names determine collection names (lowercased)
