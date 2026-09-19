"""Fail closed on missing geographical/price reference inputs for new trials."""
import hashlib
import json
from pathlib import Path

REQUIRED = {
    'hub_catalog.json':'hubs',
    'dong_centroids.json':'centroids',
    'hub_signature.json':'hubs',
    'dong_context.json':None,
    'unit_price.json':'dong_factor',
}


def inspect_references(stats_dir, require_code_geography=False):
    root=Path(stats_dir)
    result={'files':{},'limitations':[]}
    for name,key in REQUIRED.items():
        raw=(root/name).read_bytes()
        obj=json.loads(raw)
        data=obj if key is None else obj.get(key)
        if not isinstance(data,(dict,list)) or not data:
            raise ValueError(f'Missing/empty required reference: {name}:{key}')
        result['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'entries':len(data)}
        if require_code_geography and name in {'dong_centroids.json','hub_signature.json'}:
            if obj.get('_meta',{}).get('name_based_join') is not False:
                raise ValueError('Code-keyed geographical provenance required: ' + name)
        if name=='unit_price.json':
            result['absolute_category_prices_available']=bool(obj.get('l1_unit_price'))
            if not result['absolute_category_prices_available']:
                result['limitations'].append('No empirical absolute category transaction prices; historical module would use hardcoded won values. Do not label those as observed prices.')
    if not (root/'poi_menu_price.json').exists():
        result['limitations'].append('No observed individual POI menu prices. Hash price bands are synthetic, not observed shop prices.')
    return result
