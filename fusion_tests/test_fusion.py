import asyncio
import importlib.util
import json
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pygmalion_hoard.fusion.app import create_app
from pygmalion_hoard.fusion.runtime import Module

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def app(tmp_path, monkeypatch):
    for prefix in ('PYGMALION_', 'GALTON_', 'HOARD_', 'FAUSTUS_', 'HF_', 'OLLAMA_'):
        import os
        for name in list(os.environ):
            if name.startswith(prefix): monkeypatch.delenv(name)
    for name, value in {'PYGMALION_OFFLINE':'1','PYGMALION_SCHEDULER':'0','GALTON_OFFLINE':'1','GALTON_FAKE':'1','GALTON_SCHEDULER':'0','HOARD_HUB_AUTOSTART':'0'}.items():
        monkeypatch.setenv(name,value)
    return create_app(data_dir=tmp_path / 'data', port=5202)

@pytest.fixture
def client(app):
    with TestClient(app, base_url='http://127.0.0.1:5202') as client:
        client.headers.update({'Authorization':'Bearer '+app.state.token})
        yield client

def switch(client, module, enabled=True):
    response=client.put('/api/modules/'+module,json={'enabled':enabled})
    assert response.status_code == 200,response.text
    return response.json()

def call(client, name, arguments=None):
    return client.post('/api/agent/call',json={'name':name,'arguments':arguments or {}})

def test_catalog_complete_and_static_without_engines(client, app):
    from pygmalion_hoard.agent_tools import tool_catalog as training
    from galton_hoard.agent_tools import tool_catalog as evaluation
    tools=client.get('/api/agent/tools').json()['tools']
    assert len(tools)==len(training())+len(evaluation())+2==117
    assert len({t['name'] for t in tools})==117
    by_name={t['name']:t for t in tools}
    for original in training(): assert by_name[original['name']]==original
    for original in evaluation():
        name=original['name']
        if name in {t['name'] for t in training()}:name='evaluation_'+name
        assert by_name[name]=={**original,'name':name}
    assert all(m.app is None for m in app.state.modules.values())
    assert not list(app.state.data_dir.rglob('*.db'))
    assert call(client,'galton_overview').status_code==409
    assert call(client,'settings_get').status_code==409
    assert not list(app.state.data_dir.rglob('*.db'))
    assert client.get('/').status_code==200

def test_training_and_evaluation_independent_native_bridge(client, app):
    switch(client,'training')
    assert call(client,'settings_get').status_code==200
    assert call(client,'galton_overview').status_code==409
    assert client.get('/training/').status_code==200
    train=app.state.modules['training'].app.state.services
    assert train.galton.available()=={'ok':False,'via':'native','detail':''}
    switch(client,'evaluation')
    evaluate=app.state.modules['evaluation'].app.state.services
    assert evaluate.scheduler.paused()
    assert not evaluate.config.resume_pending
    assert train.config.defer_existing_jobs
    assert train.galton.call('suites_list',{})==call(client,'suites_list').json()
    assert call(client,'evaluation_settings_get').status_code==200
    assert client.get('/evaluation/').status_code==200
    assert client.get('/evaluation/api/dashboard').status_code==200
    assert client.get('/training/api/dashboard').status_code==200
    switch(client,'automation')
    assert not evaluate.scheduler.paused()
    switch(client,'automation',False)
    assert evaluate.scheduler.paused()
    switch(client,'evaluation',False)
    assert not train.galton.available()['ok']
    assert call(client,'settings_get').status_code==200
    with pytest.raises(Exception,match='Activa|refused|Galton'): train.galton.call('suites_list',{})
    switch(client,'training',False)
    assert not any(m.app is not None for m in app.state.modules.values())

def test_data_mapping_in_place_and_off_persisted(tmp_path, monkeypatch):
    data=tmp_path/'merged';training=tmp_path/'original-training';evaluation=tmp_path/'original-evaluation'
    data.mkdir();training.mkdir();evaluation.mkdir()
    (training/'sentinel.bin').write_bytes(b'train');(evaluation/'sentinel.bin').write_bytes(b'eval')
    settings={'data_paths':{'training':str(training),'evaluation':str(evaluation)},'enabled':[]}
    (data/'modules.json').write_text(json.dumps(settings))
    app=create_app(data_dir=data,port=5202)
    with TestClient(app,base_url='http://127.0.0.1:5202') as client:
        result=client.post('/api/modules/stop-all')
        assert result.status_code==200
    assert (training/'sentinel.bin').read_bytes()==b'train'
    assert (evaluation/'sentinel.bin').read_bytes()==b'eval'
    assert json.loads((data/'modules.json').read_text())['enabled']==[]
    assert list(training.iterdir())==[training/'sentinel.bin']
    assert list(evaluation.iterdir())==[evaluation/'sentinel.bin']

@pytest.mark.asyncio
async def test_stopping_drains_native_calls_before_database_close(monkeypatch,tmp_path):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from galton_hoard import agent_tools
    entered=threading.Event();release=threading.Event();closed=[]
    @asynccontextmanager
    async def lifespan(app):
        app.state.services=object()
        yield
        assert release.is_set();closed.append(True)
    native=FastAPI(lifespan=lifespan)
    module=Module({'id':'evaluation'},tmp_path,lambda *args:native)
    def slow(*args,**kwargs):entered.set();release.wait(3);return {'ok':True}
    monkeypatch.setattr(agent_tools,'call_tool',slow)
    await module.set_enabled(True)
    thread=threading.Thread(target=lambda:module.native_call('slow',{}));thread.start()
    assert await asyncio.to_thread(entered.wait,1)
    stopping=asyncio.create_task(module.set_enabled(False));await asyncio.sleep(.05)
    assert module.stopping and not stopping.done() and not closed
    with pytest.raises(Exception):module.native_call('new',{})
    release.set();await stopping;thread.join()
    assert closed==[True] and module.app is None

def test_stdio_catalog_has_no_engine_import_side_effects(tmp_path):
    import subprocess,sys,os
    code='import mcp_server,sys; print(len(mcp_server.catalog())); assert "galton_hoard.services" not in sys.modules; assert "pygmalion_hoard.services" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30,env={**os.environ,'PYGMALION_DATA_DIR':str(tmp_path)})
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='117'
    assert list(tmp_path.iterdir())==[]

@pytest.mark.asyncio
async def test_retry_pending_shutdown_keeps_database_open_until_workers_finish(tmp_path):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    class Services:
        attempts=0
        closed=False
        def stop(self):
            self.attempts+=1
            if self.attempts==1:raise RuntimeError('worker still joining')
            self.closed=True
    svc=Services()
    @asynccontextmanager
    async def life(app):
        app.state.services=svc
        yield
        svc.stop()
    native=FastAPI(lifespan=life)
    module=Module({'id':'evaluation'},tmp_path,lambda *args:native)
    await module.set_enabled(True)
    with pytest.raises(RuntimeError):await module.set_enabled(False)
    assert module.app is native and module.cleanup_pending and not svc.closed
    await module.set_enabled(False)
    assert module.app is None and svc.closed and svc.attempts==2

def test_native_contract_clock_and_validation(client, app):
    switch(client,'training')
    bridge=app.state.modules['training'].app.state.services.galton
    assert isinstance(bridge.clock(),float)
    bridge.sleep(0)
    assert call(client,'pygmalion_set_module',{'module':'training','enabled':'yes'}).status_code==400
    assert client.post('/api/agent/call',json=[]).status_code==400
    assert client.post('/api/agent/call',content='{oops').status_code==400

def test_adoption_maps_original_data_without_opening_it(tmp_path):
    import subprocess,sys
    evaluation=tmp_path/'original';evaluation.mkdir()
    sentinel=evaluation/'galton.db';sentinel.write_bytes(b'not a real database')
    data=tmp_path/'merged'
    run=subprocess.run([sys.executable,str(ROOT/'scripts/adopt_evaluation.py'),'--data-dir',str(data),'--evaluation-data',str(evaluation)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    settings=json.loads((data/'modules.json').read_text())
    assert settings['enabled']==[] and settings['data_paths']['evaluation']==str(evaluation.resolve())
    assert sentinel.read_bytes()==b'not a real database' and list(evaluation.iterdir())==[sentinel]

def test_all_historical_queued_jobs_wait_for_explicit_resume(client,app):
    switch(client,'training')
    svc=app.state.modules['training'].app.state.services
    with svc.db.tx():
        previous=[svc.store.create_job('convert','cpu',{}) for _ in range(2001)]
    svc.jobs.start()
    assert len(svc.jobs._held_jobs)==2001
    assert svc.store.next_queued('cpu',exclude_ids=svc.jobs._held_jobs) is None
    fresh=svc.store.create_job('convert','cpu',{})
    assert svc.store.next_queued('cpu',exclude_ids=svc.jobs._held_jobs)['id']==fresh['id']
    assert svc.jobs.view(previous[0])['resumable']
    resumed=svc.jobs.resume(previous[0]['id'])
    assert resumed['state']=='queued' and previous[0]['id'] not in svc.jobs._held_jobs

def test_training_evaluator_finishes_against_native_evaluation_engine(client,app):
    from evaluation_tests.helpers import add_gguf
    switch(client,'training');switch(client,'evaluation')
    training=app.state.modules['training'].app.state.services
    evaluation=app.state.modules['evaluation'].app.state.services
    add_gguf(evaluation,'fusion-parent',accuracy=.5,seed=1)
    add_gguf(evaluation,'fusion-child',accuracy=1)
    parent=training.store.create_artifact('ollama','fusion-parent',path='fusion-parent')
    child=training.store.create_artifact('ollama','fusion-child',path='fusion-child',parents=[parent['id']])
    result=training.evaluator.run(child['id'],against=parent['id'],intent='smoke',suites=['rapida'],regression=False,settings={'effort':'off'})
    assert result['run'] and result['contestants']['parent']!=result['contestants']['child']
    assert result['verdict'] in ('better','no_clear_difference')
    assert evaluation.store.run(result['run'])['state']=='done'
    assert evaluation.store.run(result['run'])['caller']=='pygmalion'
    assert training.store.artifact(child['id'])['metrics']['galton']['run']==result['run']
    assert evaluation.fake_world.calls


def test_native_errors_keep_the_training_evaluator_contract(client,app):
    from pygmalion_hoard.errors import PygmalionError
    switch(client,'training');switch(client,'evaluation')
    bridge=app.state.modules['training'].app.state.services.galton
    with pytest.raises(PygmalionError) as caught:bridge.call('model_get',{'model':'missing-native'})
    assert caught.value.code=='galton_unavailable' and 'missing-native' in caught.value.message
