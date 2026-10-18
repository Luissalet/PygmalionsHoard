"""Generate the combined static catalogue without creating any Services object."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pygmalion_hoard.agent_tools import tool_catalog as training
from galton_hoard.agent_tools import tool_catalog as evaluation


def main():
    tools=[];names=set()
    for module,source in [('training',training()),('evaluation',evaluation())]:
        for item in source:
            item={**item,'module':module}
            if item['name'] in names:
                item['native_name']=item['name'];item['name']='evaluation_'+item['name']
            names.add(item['name']);tools.append(item)
    text=json.dumps(tools,indent=2,ensure_ascii=False)
    target=ROOT/'pygmalion_hoard/fusion/catalog.json'
    if '--check' in sys.argv:
        assert target.read_text(encoding='utf-8')==text,'Combined catalogue is stale'
    else:target.write_text(text,encoding='utf-8')
    print(f'{len(tools)} native tools; static catalogue is current')
    return 0

if __name__=='__main__':raise SystemExit(main())
