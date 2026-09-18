import importlib.util
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lvfi", ROOT / "frame_interpolation.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def linear_backend(frames, multiplier):
    out=[]
    for i in range(frames.shape[0]-1):
        a=frames[i:i+1]
        b=frames[i+1:i+2]
        out.append(a)
        for j in range(1,multiplier):
            t=j/multiplier
            out.append(a*(1-t)+b*t)
    out.append(frames[-1:])
    return torch.cat(out, dim=0)


def seq(vals):
    return torch.tensor(vals, dtype=torch.float32).view(-1,1,1,1).expand(-1,1,1,3).clone()


def test_multiplier_resolution():
    assert mod.resolve_integer_multiplier(24,120)==(5,120)
    assert mod.resolve_integer_multiplier(30,120)==(4,120)
    assert mod.resolve_integer_multiplier(60,120)==(2,120)
    m,f=mod.resolve_integer_multiplier(23.976,120)
    assert m==5 and abs(f-119.88)<1e-6
    assert mod.resolve_integer_multiplier(119.88,120)==(1,119.88)
    assert mod.resolve_integer_multiplier(120.0,120)==(1,120.0)
    assert mod.resolve_integer_multiplier(120.006018054,120)==(1,120.006018054)
    try:
        mod.resolve_integer_multiplier(100.0,120.0)
    except ValueError as exc:
        assert "not close enough" in str(exc)
    else:
        raise AssertionError("100 fps must not silently bypass a 120-fps target")


def test_two_batches_equal_one_shot():
    state=mod.LongVideoInterpolationState(signature=("x",))
    a=seq([0,1,2,3])
    b=seq([4,5,6])
    oa=mod.stitch_batch_with_backend(a,state,4,linear_backend)
    ob=mod.stitch_batch_with_backend(b,state,4,linear_backend)
    stitched=torch.cat((oa,ob),dim=0)
    reference=linear_backend(torch.cat((a,b),dim=0),4)
    assert stitched.shape==reference.shape
    assert torch.equal(stitched,reference)
    # The boundary pair 3->4 must contain 3.25, 3.5, 3.75.
    vals=stitched[:,0,0,0]
    idx=(vals==3).nonzero().item()
    assert torch.allclose(vals[idx+1:idx+4], torch.tensor([3.25,3.5,3.75]))


def test_24fps_output_count_across_44_frame_boundary():
    state=mod.LongVideoInterpolationState(signature=("x",))
    a=seq(list(range(44)))
    b=seq(list(range(44,88)))
    oa=mod.stitch_batch_with_backend(a,state,5,linear_backend)
    ob=mod.stitch_batch_with_backend(b,state,5,linear_backend)
    assert oa.shape[0]==216  # (44-1)*5+1
    assert ob.shape[0]==220  # 44 newly arriving source intervals * 5
    assert oa.shape[0]+ob.shape[0]==(88-1)*5+1


def test_scene_hold_count():
    x=seq([0])
    h=mod._hold_intermediates(x,4,torch.float16)
    assert h.shape[0]==4
    assert h.dtype==torch.float16
    assert torch.all(h==0)

def test_split_eyes_core_with_fake_model():
    import types, sys
    comfy = types.ModuleType('comfy')
    mm = types.ModuleType('comfy.model_management')
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get('comfy')
    old_mm = sys.modules.get('comfy.model_management')
    sys.modules['comfy'] = comfy
    sys.modules['comfy.model_management'] = mm

    class FakeNet:
        pad_align = 1
        def memory_used_forward(self, shape, dtype): return 0
        def __call__(self, a, b, timestep, cache=None):
            # timestep is [B,1,H,W]
            return a * (1 - timestep) + b * timestep
    class FakePatcher:
        load_device = torch.device('cpu')
        def __init__(self): self.model = FakeNet()
        def model_dtype(self): return torch.float32

    try:
        # 2 temporal frames, each SBS eye is width=2. Left eye 0->1, right 10->11.
        f0 = torch.tensor([[[[0.,0.,0.],[0.,0.,0.],[.10,.10,.10],[.10,.10,.10]]]])
        f1 = torch.tensor([[[[1.,1.,1.],[1.,1.,1.],[.20,.20,.20],[.20,.20,.20]]]])
        images = torch.cat((f0,f1), dim=0)
        out = mod.interpolate_sequence_core(FakePatcher(), images, 4, stereo_mode='split_eyes', scene_cut=False, cpu_output='float32', timestep_batch=2)
        assert out.shape == (5,1,4,3)
        assert torch.allclose(out[:,0,0,0], torch.tensor([0.,.25,.5,.75,1.]))
        assert torch.allclose(out[:,0,2,0], torch.tensor([.10,.125,.15,.175,.20]))
    finally:
        if old_comfy is None: sys.modules.pop('comfy', None)
        else: sys.modules['comfy'] = old_comfy
        if old_mm is None: sys.modules.pop('comfy.model_management', None)
        else: sys.modules['comfy.model_management'] = old_mm


if __name__=='__main__':
    for fn in [test_multiplier_resolution,test_two_batches_equal_one_shot,test_24fps_output_count_across_44_frame_boundary,test_scene_hold_count,test_split_eyes_core_with_fake_model]:
        fn()
    print('frame interpolation state tests: PASS')


def test_streaming_exact_duration_and_bounded_chunks():
    import types, sys
    comfy = types.ModuleType('comfy')
    mm = types.ModuleType('comfy.model_management')
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get('comfy')
    old_mm = sys.modules.get('comfy.model_management')
    sys.modules['comfy'] = comfy
    sys.modules['comfy.model_management'] = mm

    class FakeNet:
        pad_align = 1
        def memory_used_forward(self, shape, dtype): return 0
        def __call__(self, a, b, timestep, cache=None):
            return a * (1 - timestep) + b * timestep
    class FakePatcher:
        load_device = torch.device('cpu')
        _longvideo_model_name = 'rife_v4.25.safetensors'
        def __init__(self): self.model = FakeNet()
        def model_dtype(self): return torch.float32

    try:
        patcher=FakePatcher()
        state=mod.LongVideoInterpolationState(signature=('stream',))
        key=(123,'node')
        mod._STATES[key]=state
        a=seq([0.0,0.1,0.2,0.3])
        b=seq([0.4,0.5,0.6])
        sa=mod.LongVideoInterpolationStreamBatch(
            patcher,a,state,key,4,30.0,120.0,'full_sbs',False,0.22,'float32',1,1,5,'sync','sync',False)
        ca=list(sa.iter_chunks())
        assert all(x.shape[0] <= 5 for x in ca)
        assert sum(x.shape[0] for x in ca)==13
        assert state.input_frames==4 and state.output_frames==13

        sb=mod.LongVideoInterpolationStreamBatch(
            patcher,b,state,key,4,30.0,120.0,'full_sbs',False,0.22,'float32',1,1,5,'sync','sync',True)
        cb=list(sb.iter_chunks())
        assert all(x.shape[0] <= 5 for x in cb)
        assert sum(x.shape[0] for x in cb)==15
        stitched=torch.cat(ca+cb, dim=0)
        reference=linear_backend(torch.cat((a,b),dim=0),4)
        reference=torch.cat((reference, b[-1:].expand(3,-1,-1,-1).clone()),dim=0)
        assert stitched.shape[0]==7*4==28
        assert torch.equal(stitched,reference)
        assert state.input_frames==7 and state.output_frames==28
        assert 'duration_change=+0.000000000%' in sb.summary
        assert key not in mod._STATES
    finally:
        mod._STATES.pop((123,'node'),None)
        if old_comfy is None: sys.modules.pop('comfy', None)
        else: sys.modules['comfy'] = old_comfy
        if old_mm is None: sys.modules.pop('comfy.model_management', None)
        else: sys.modules['comfy.model_management'] = old_mm


def test_streaming_target_rate_bypass_keeps_frames_and_duration():
    class DummyPatcher:
        _longvideo_model_name = "rife_v4.25.safetensors"

    state=mod.LongVideoInterpolationState(signature=("bypass",))
    key=(456,"bypass-node")
    mod._STATES[key]=state
    a=seq([0.0,0.1,0.2,0.3])
    b=seq([0.4,0.5,0.6])
    try:
        sa=mod.LongVideoInterpolationStreamBatch(
            DummyPatcher(),a,state,key,1,119.88,119.88,'full_sbs',False,0.22,'float32',1,1,2,'sync','sync',False)
        ca=list(sa.iter_chunks())
        assert [int(x.shape[0]) for x in ca] == [2,2]
        assert torch.equal(torch.cat(ca,dim=0), a)
        assert state.input_frames==4 and state.output_frames==4
        assert sa.stats.get('bypass_active') is True
        assert sa.stats.get('rife_forward_calls') == 0

        sb=mod.LongVideoInterpolationStreamBatch(
            DummyPatcher(),b,state,key,1,119.88,119.88,'full_sbs',False,0.22,'float32',1,1,2,'sync','sync',True)
        cb=list(sb.iter_chunks())
        assert [int(x.shape[0]) for x in cb] == [2,1]
        assert torch.equal(torch.cat(cb,dim=0), b)
        assert state.input_frames==7 and state.output_frames==7
        assert 'BYPASS 1x' in sb.summary
        assert 'duration_change=+0.000000000%' in sb.summary
        assert key not in mod._STATES
    finally:
        mod._STATES.pop(key,None)


def test_stream_node_uses_target_rate_bypass_for_11988_to_120():
    import types
    class DummyPatcher:
        _longvideo_model_name = "rife_v4.25.safetensors"
    session = types.SimpleNamespace(source_name="already120.mp4", finished=True)
    images = seq([0.0,0.1,0.2])
    node = mod.LV_LongVideoFrameInterpolationStream()
    stream, fps, status = node.make_stream(
        DummyPatcher(), images, session, 119.88, 120.0, "full_sbs",
        True, 0.22, "float32", 1, 1, 8, "sync", "sync", unique_id="target-bypass"
    )
    chunks=list(stream.iter_chunks())
    assert fps == 119.88
    assert 'BYPASS 1x' in status
    assert torch.equal(torch.cat(chunks,dim=0), images)
    assert stream.stats.get('rife_forward_calls') == 0


def test_scene_proxy_obvious_cut_and_same_frame():
    a = torch.zeros(1, 216, 384, 3, dtype=torch.float16)
    b = a.clone()
    c = torch.ones_like(a)
    assert mod._scene_cut_pair(a, b, 0.22) is False
    assert mod._scene_cut_pair(a, c, 0.22) is True


def test_stream_carries_source_name_from_state():
    class DummyPatcher:
        _longvideo_model_name = "rife_v4.25.safetensors"
    state = mod.LongVideoInterpolationState(signature=("naming",))
    state.source_name = "Folder/My Movie.mkv"
    images = torch.zeros(2, 4, 8, 3)
    stream = mod.LongVideoInterpolationStreamBatch(
        DummyPatcher(), images, state, (1,"n"), 2, 60.0, 120.0,
        "full_sbs", False, 0.22, "float16", 1, 1, 8, "sync", "sync", False
    )
    assert stream.source_name == "Folder/My Movie.mkv"


def test_make_stream_copies_vda_session_source_name():
    import types
    class DummyPatcher:
        _longvideo_model_name = "rife_v4.25.safetensors"
    session = types.SimpleNamespace(source_name="espresso.mp4", finished=False)
    images = torch.zeros(2, 4, 8, 3)
    node = mod.LV_LongVideoFrameInterpolationStream()
    stream, fps, status = node.make_stream(
        DummyPatcher(), images, session, 60.0, 120.0, "full_sbs",
        True, 0.22, "float16", 1, 1, 8, "sync", "sync", unique_id="source-copy"
    )
    try:
        assert stream.source_name == "espresso.mp4"
        assert fps == 120.0
    finally:
        mod._STATES.pop((id(session), "source-copy"), None)


def test_scene_proxy_preserves_old_area_metric_semantics():
    torch.manual_seed(425)
    a = torch.rand(1, 216, 384, 3, dtype=torch.float16)
    b = torch.rand(1, 216, 384, 3, dtype=torch.float16)
    def old_metric(a, b):
        x = a.float().movedim(-1, 1)
        y = b.float().movedim(-1, 1)
        x = x[:,0:1]*0.2126 + x[:,1:2]*0.7152 + x[:,2:3]*0.0722
        y = y[:,0:1]*0.2126 + y[:,1:2]*0.7152 + y[:,2:3]*0.0722
        x = torch.nn.functional.interpolate(x, size=(48,64), mode="area")
        y = torch.nn.functional.interpolate(y, size=(48,64), mode="area")
        return torch.mean(torch.abs(x-y)).item()
    x = torch.nn.functional.adaptive_avg_pool2d(a.movedim(-1,1), (48,64)).float()
    y = torch.nn.functional.adaptive_avg_pool2d(b.movedim(-1,1), (48,64)).float()
    x = x[:,0:1]*0.2126 + x[:,1:2]*0.7152 + x[:,2:3]*0.0722
    y = y[:,0:1]*0.2126 + y[:,1:2]*0.7152 + y[:,2:3]*0.0722
    new_metric = torch.mean(torch.abs(x-y)).item()
    assert abs(new_metric - old_metric(a,b)) < 5e-4


def test_dev24_transfer_mode_ui_and_cpu_fallback():
    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert req["d2h_mode"][0] == ["sync", "async_pinned"]
    assert req["d2h_mode"][1]["default"] == "async_pinned"
    assert req["h2d_mode"][0] == ["sync", "async_pinned"]
    assert req["h2d_mode"][1]["default"] == "async_pinned"
    assert req["pair_batch"][1]["default"] == 1

    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    class FakeNet:
        pad_align = 1
        def memory_used_forward(self, shape, dtype): return 0
        def __call__(self, a, b, timestep, cache=None): return a*(1-timestep)+b*timestep
    class FakePatcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self): self.model = FakeNet()
        def model_dtype(self): return torch.float32
    try:
        stats = {}
        chunks = list(mod.iter_interpolated_sequence_chunks(
            FakePatcher(), seq([0., 1.]), 2, scene_cut=False, cpu_output="float32",
            chunk_frames=8, d2h_mode="async_pinned", h2d_mode="async_pinned", stats=stats
        ))
        assert sum(int(x.shape[0]) for x in chunks) == 3
        assert stats["d2h_mode_requested"] == "async_pinned"
        assert stats["d2h_mode_active"] == "sync"
        assert stats["h2d_mode_requested"] == "async_pinned"
        assert stats["h2d_mode_active"] == "sync"
        assert "layout_seconds" in stats and "h2d_seconds" in stats and "h2d_wait_seconds" in stats
    finally:
        if old_comfy is None: sys.modules.pop("comfy", None)
        else: sys.modules["comfy"] = old_comfy
        if old_mm is None: sys.modules.pop("comfy.model_management", None)
        else: sys.modules["comfy.model_management"] = old_mm


def test_dev231_pinned_pool_reuses_buffers_without_reallocating():
    # Exercise pool bookkeeping without requiring a CUDA device in CI.
    orig_empty = mod.torch.empty
    try:
        def fake_empty(shape, dtype=None, device=None, pin_memory=False):
            assert pin_memory is True
            return orig_empty(shape, dtype=dtype, device='cpu')
        mod.torch.empty = fake_empty
        pool = mod._PinnedCpuBufferPool()
        a = pool.acquire((1,3,4,5), torch.float16)
        assert pool.allocations == 1 and pool.reuses == 0 and pool.inflight == 1
        pool.release(a)
        b = pool.acquire((1,3,4,5), torch.float16)
        assert b.data_ptr() == a.data_ptr()
        assert pool.allocations == 1 and pool.reuses == 1 and pool.inflight == 1
        pool.release(b)
        assert pool.inflight == 0 and pool.peak_inflight == 1
    finally:
        mod.torch.empty = orig_empty


def test_dev231_new_session_retires_old_interpolation_state_and_pool():
    class DummyPool:
        def __init__(self): self.cleared=False
        def clear(self): self.cleared=True
    class DummyPatcher:
        _longvideo_model_name = 'rife_v4.25.safetensors'
    import types
    old_session = types.SimpleNamespace(source_name='old.mp4', finished=False)
    new_session = types.SimpleNamespace(source_name='new.mp4', finished=False)
    old_key=(id(old_session),'same-node')
    old=mod.LongVideoInterpolationState(signature=('old',))
    old.pinned_pool=DummyPool()
    old.h2d_pinned_pool=DummyPool()
    old_h2d = old.h2d_pinned_pool
    mod._STATES[old_key]=old
    images=torch.zeros(2,4,8,3)
    node=mod.LV_LongVideoFrameInterpolationStream()
    stream,_,_=node.make_stream(
        DummyPatcher(), images, new_session, 60.0, 120.0, 'full_sbs',
        True, 0.22, 'float16', 1, 1, 8, 'async_pinned', 'async_pinned', unique_id='same-node'
    )
    try:
        assert old_key not in mod._STATES
        assert old.pinned_pool is None
        assert old.h2d_pinned_pool is None
        assert old_h2d.cleared is True
        assert stream.state_key == (id(new_session),'same-node')
    finally:
        mod._STATES.pop((id(new_session),'same-node'),None)


def test_dev241_scene_prepass_preserves_pairwise_decisions():
    torch.manual_seed(241)
    images = torch.rand(6, 72, 128, 3, dtype=torch.float16)
    # Force one obvious hard cut while leaving other pairs natural.
    images[3].zero_()
    images[4].fill_(1.0)
    expected = [mod._scene_cut_pair(images[i:i+1], images[i+1:i+2], 0.22) for i in range(5)]
    got, seconds = mod._precompute_scene_flags(images, 0.22)
    assert got == expected
    assert seconds >= 0.0


def test_dev241_scene_prepass_disabled_threshold():
    images = torch.rand(4, 16, 16, 3, dtype=torch.float16)
    got, seconds = mod._precompute_scene_flags(images, 0.0)
    assert got == [False, False, False]
    assert seconds == 0.0



def test_dev25_pair_batch_matches_pair1_for_2x_and_reduces_calls():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    class FakeNet:
        pad_align = 1
        def memory_used_forward(self, shape, dtype): return 0
        def __call__(self, a, b, timestep, cache=None): return a*(1-timestep)+b*timestep
    class FakePatcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self): self.model = FakeNet()
        def model_dtype(self): return torch.float32
    try:
        images = seq([0., .1, .2, .3, .4, .5])
        s1, s2 = {}, {}
        o1 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            FakePatcher(), images, 2, scene_cut=False, cpu_output="float32",
            timestep_batch=1, pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync", stats=s1
        )), dim=0)
        o2 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            FakePatcher(), images, 2, scene_cut=False, cpu_output="float32",
            timestep_batch=1, pair_batch=2, chunk_frames=64, d2h_mode="sync", h2d_mode="sync", stats=s2
        )), dim=0)
        assert torch.equal(o1, o2)
        assert s1["rife_forward_calls"] == 5
        assert s2["rife_forward_calls"] == 3
        assert s2["rife_pairs_submitted"] == 5
        assert s2["pair_batch_active"] == 2
    finally:
        if old_comfy is None: sys.modules.pop("comfy", None)
        else: sys.modules["comfy"] = old_comfy
        if old_mm is None: sys.modules.pop("comfy.model_management", None)
        else: sys.modules["comfy.model_management"] = old_mm


def test_dev25_pair_batch_preserves_scene_cut_hold_order():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    class FakeNet:
        pad_align = 1
        def memory_used_forward(self, shape, dtype): return 0
        def __call__(self, a, b, timestep, cache=None): return a*(1-timestep)+b*timestep
    class FakePatcher:
        load_device = torch.device("cpu")
        def __init__(self): self.model = FakeNet()
        def model_dtype(self): return torch.float32
    try:
        images = seq([0., .1, 1.0, .9])
        out = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            FakePatcher(), images, 2, scene_cut=True, scene_threshold=.5, cpu_output="float32",
            pair_batch=2, chunk_frames=64, d2h_mode="sync", h2d_mode="sync"
        )), dim=0)
        vals = out[:,0,0,0]
        # 0->.1 interpolates; .1->1.0 is a cut, so midpoint holds .1; 1.0->.9 interpolates.
        assert torch.allclose(vals, torch.tensor([0., .05, .1, .1, 1., .95, .9]))
    finally:
        if old_comfy is None: sys.modules.pop("comfy", None)
        else: sys.modules["comfy"] = old_comfy
        if old_mm is None: sys.modules.pop("comfy.model_management", None)
        else: sys.modules["comfy.model_management"] = old_mm


def test_dev26_ifnet_stage_profiler_cpu_preserves_output_and_counts():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

    class Encode(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 0.01

    class ToyIFNet(torch.nn.Module):
        pad_align = 1
        def __init__(self):
            super().__init__()
            self.encode = Encode()
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        def memory_used_forward(self, shape, dtype):
            return 0
        def forward(self, a, b, timestep, cache=None):
            # Exercise the same high-level IFNet shape: two encode calls followed
            # by ordered block stages. The encode result is intentionally not used
            # in the toy blend; this test verifies instrumentation, not RIFE math.
            self.encode(a)
            self.encode(b)
            x = (a * (1 - timestep) + b * timestep) * self.scale
            for block in self.blocks:
                x = block(x)
            return x

    class Patcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self):
            self.model = ToyIFNet()
        def model_dtype(self):
            return torch.float32

    try:
        images = torch.rand(4, 8, 8, 3)
        s_off, s_on = {}, {}
        out_off = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            Patcher(), images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="off", stats=s_off,
        )), dim=0)
        out_on = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            Patcher(), images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="stages", stats=s_on,
        )), dim=0)
        assert torch.equal(out_off, out_on)
        assert s_off["ifnet_profile_active"] == "off"
        assert s_on["ifnet_profile_active"] == "stages"
        assert s_on["ifnet_stage_names"] == ["encode", "block0", "block1"]
        assert s_on["ifnet_stage_calls"]["encode"] == 6
        assert s_on["ifnet_stage_calls"]["block0"] == 3
        assert s_on["ifnet_stage_calls"]["block1"] == 3
        assert s_on["ifnet_stage_profiled_forwards"] == 3
        assert s_on["ifnet_stage_unattributed_seconds"] >= 0.0
        req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
        assert req["ifnet_profile"][0] == ["off", "stages", "detail", "warp_detail", "block4_detail", "block4_res_inplace", "block4_scale1_bypass", "conv_family_detail", "outer_detail", "outer_cat_pushdown", "warp_postopt_detail"]
        assert req["ifnet_profile"][1]["default"] == "off"
    finally:
        if old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = old_comfy
        if old_mm is None:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = old_mm


def test_dev27_adjacent_feature_cache_preserves_output_and_reduces_encode_calls():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

    class Encode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x[:, :1] * 0.5

    class ToyCachedIFNet(torch.nn.Module):
        pad_align = 1
        def __init__(self):
            super().__init__()
            self.encode = Encode()
            self.blocks = torch.nn.ModuleList([])
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        def memory_used_forward(self, shape, dtype):
            return 0
        def extract_features(self, img):
            return self.encode(img)
        def forward(self, a, b, timestep, cache=None):
            B = a.shape[0]
            # Match current ComfyUI IFNet cache semantics: cached head features
            # are expanded across the active inference batch and therefore avoid
            # calling encode() inside forward.
            if cache and "img0" in cache:
                _ = cache["img0"].expand(B, -1, -1, -1)
            else:
                _ = self.encode(a)
            if cache and "img1" in cache:
                _ = cache["img1"].expand(B, -1, -1, -1)
            else:
                _ = self.encode(b)
            return (a * (1 - timestep) + b * timestep) * self.scale

    class Patcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self):
            self.model = ToyCachedIFNet()
        def model_dtype(self):
            return torch.float32

    try:
        images = torch.rand(4, 8, 8, 3)
        p_off, p_on = Patcher(), Patcher()
        s_off, s_on = {}, {}
        out_off = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            p_off, images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="off", feature_cache="off", stats=s_off,
        )), dim=0)
        out_on = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            p_on, images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="stages", feature_cache="adjacent", stats=s_on,
        )), dim=0)
        assert torch.equal(out_off, out_on)
        assert p_off.model.encode.calls == 6  # 2 head calls * 3 temporal pairs
        assert p_on.model.encode.calls == 4   # each of 4 source frames encoded once
        assert s_on["feature_cache_active"] == "adjacent"
        assert s_on["feature_cache_extract_calls"] == 4
        assert s_on["feature_cache_reuses"] == 2
        assert s_on["feature_cache_pairs"] == 3
        assert s_on["ifnet_stage_calls"]["encode"] == 4
        req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
        assert req["feature_cache"][0] == ["off", "adjacent"]
        assert req["feature_cache"][1]["default"] == "adjacent"
    finally:
        if old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = old_comfy
        if old_mm is None:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = old_mm



def test_dev28_detail_profiler_splits_image_and_feature_warps_without_math_change():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

    class Encode(torch.nn.Module):
        def forward(self, x):
            return x[:, 0:1].repeat(1, 4, 1, 1)

    class ToyWarpIFNet(torch.nn.Module):
        pad_align = 1
        def __init__(self):
            super().__init__()
            self.encode = Encode()
            self.blocks = torch.nn.ModuleList([])
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        def memory_used_forward(self, shape, dtype):
            return 0
        def extract_features(self, img):
            return self.encode(img)
        def warp(self, img, flow):
            # Identity math, but exercise both 3-channel image and 4-channel
            # feature warp call sites exactly as the detail wrapper sees them.
            return img + flow[:, :1] * 0.0
        def forward(self, a, b, timestep, cache=None):
            B = a.shape[0]
            f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else self.encode(a)
            z = torch.zeros((B, 2, a.shape[2], a.shape[3]), dtype=a.dtype, device=a.device)
            _ = self.warp(a, z)
            _ = self.warp(f0, z)
            return (a * (1 - timestep) + b * timestep) * self.scale

    class Patcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self):
            self.model = ToyWarpIFNet()
        def model_dtype(self):
            return torch.float32

    try:
        images = torch.rand(4, 8, 8, 3)
        p0, p1 = Patcher(), Patcher()
        s0, s1 = {}, {}
        out0 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            p0, images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="off", feature_cache="adjacent", stats=s0,
        )), dim=0)
        out1 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            p1, images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="detail", feature_cache="adjacent", stats=s1,
        )), dim=0)
        assert torch.equal(out0, out1)
        assert s1["ifnet_profile_active"] == "detail"
        assert s1["ifnet_stage_calls"]["warp_image"] == 3
        assert s1["ifnet_stage_calls"]["warp_feature"] == 3
        assert s1["ifnet_stage_calls"].get("warp_other", 0) == 0
        assert s1["feature_cache_extract_calls"] == 4
        assert "warp" not in p1.model.__dict__, "detail profiler did not restore class-bound warp method"
    finally:
        if old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = old_comfy
        if old_mm is None:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = old_mm


def test_dev29_warp_inner_profiler_exact_math_and_stages():
    class FakeWarpNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._warp_grids = {}
        def build(self, H, W, device):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

    net = FakeWarpNet()
    net.build(8, 10, torch.device("cpu"))
    img = torch.rand(1, 3, 8, 10, dtype=torch.float32)
    flow = torch.randn(1, 2, 8, 10, dtype=torch.float32) * 0.25
    original_bound = net.warp
    expected = original_bound(img, flow)
    prof = mod._IFNetStageProfiler(net, profile_warp_inner=True)
    try:
        assert prof.usable
        prof.begin("cpu")
        actual = net.warp(img, flow)
        trace = prof.end()
        assert torch.equal(actual, expected)
        for part in ("norm", "grid", "cast_in", "sample", "cast_out"):
            spans = trace["events"].get(f"warp_image_{part}", [])
            assert len(spans) == 1
        assert not trace["events"].get("warp_image_fallback")
    finally:
        prof.close()
    # close() must restore the class method rather than leaving an instance wrapper.
    assert "warp" not in net.__dict__
    assert torch.equal(net.warp(img, flow), expected)



def test_dev210_warp_input_fp32_reuse_preserves_exact_math_and_reuses_casts():
    class FakeWarpNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._warp_grids = {}
        def build(self, H, W, device):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

    net = FakeWarpNet()
    net.build(8, 10, torch.device("cpu"))
    # float64 forces the same explicit ->float32->source-dtype round trip that
    # production fp16 uses, while CPU grid_sample itself remains supported.
    img0 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    img1 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    flow = torch.randn(1, 2, 8, 10, dtype=torch.float32) * 0.25
    expected0 = [net.warp(img0, flow) for _ in range(5)]
    expected1 = [net.warp(img1, flow) for _ in range(4)]
    opt = mod._IFNetWarpInputCache(net)
    try:
        opt.begin_forward()
        actual0 = [net.warp(img0, flow) for _ in range(5)]
        actual1 = [net.warp(img1, flow) for _ in range(4)]
        opt.end_forward()
        assert all(torch.equal(a, b) for a, b in zip(actual0, expected0))
        assert all(torch.equal(a, b) for a, b in zip(actual1, expected1))
        snap = opt.snapshot()
        assert snap["casts"] == 2
        assert snap["reuses"] == 7
        assert snap["peak_entries"] == 2
        assert snap["forwards"] == 1
        assert snap["fallbacks"] == 0

        # Cache lifetime is one IFNet forward: same source must be converted again.
        opt.begin_forward()
        again = net.warp(img0, flow)
        opt.end_forward()
        assert torch.equal(again, expected0[0])
        snap = opt.snapshot()
        assert snap["casts"] == 3
        assert snap["forwards"] == 2
    finally:
        opt.close()
    assert "warp" not in net.__dict__
    assert torch.equal(net.warp(img0, flow), expected0[0])

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert req["warp_input_cache"][0] == ["off", "fp32_reuse", "grid_reuse"]
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"


def test_dev2101_warp_input_stats_are_live_before_generator_finalizer():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

    class ToyWarpIFNet(torch.nn.Module):
        pad_align = 1
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self._warp_grids = {}
        def memory_used_forward(self, shape, dtype):
            return 0
        def _build(self, H, W, device):
            if (H, W) in self._warp_grids:
                return
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)
        def forward(self, a, b, timestep, cache=None):
            self._build(a.shape[2], a.shape[3], a.device)
            flow = torch.zeros((a.shape[0], 2, a.shape[2], a.shape[3]), device=a.device, dtype=a.dtype)
            # Two unique sources, four total warps => two casts + two reuses per forward.
            wa0 = self.warp(a, flow)
            wa1 = self.warp(a, flow)
            wb0 = self.warp(b, flow)
            wb1 = self.warp(b, flow)
            return (wa0 + wa1 + wb0 + wb1) * 0.25

    class Patcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self):
            self.model = ToyWarpIFNet()
        def model_dtype(self):
            return torch.float32

    try:
        images = torch.rand(4, 8, 8, 3)
        stats = {}
        chunks = mod.iter_interpolated_sequence_chunks(
            Patcher(), images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=1, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="off", feature_cache="off", warp_input_cache="fp32_reuse", stats=stats,
        )
        first = next(chunks)
        assert first.shape[0] == 1
        # include_first is yielded before model setup. Advance once more so the
        # first temporal pair has executed, but the generator is still not exhausted.
        second = next(chunks)
        assert second.shape[0] == 1
        # Counter publication must happen immediately after the first IFNet forward,
        # before the generator is exhausted/closed.
        assert stats["warp_input_cache_active"] == "fp32_reuse"
        assert stats["warp_input_forwards"] >= 1
        assert stats["warp_input_casts"] >= 2
        assert stats["warp_input_reuses"] >= 2
        list(chunks)
        assert stats["warp_input_fallbacks"] == 0
    finally:
        if old_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = old_comfy
        if old_mm is None:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = old_mm


def test_dev211_warp_grid_flow_reuse_preserves_exact_math_and_reuses_stage_grids():
    class FakeWarpNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._warp_grids = {}
        def build(self, H, W, device):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

    net = FakeWarpNet()
    net.build(8, 10, torch.device("cpu"))
    img0 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    img1 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    f0 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    f1 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    base_flow = torch.randn(1, 4, 8, 10, dtype=torch.float32) * 0.2
    deltas = [torch.randn_like(base_flow) * 0.03 for _ in range(4)]

    def sequence():
        flow = base_flow.clone()
        outs = []
        # Stage 0 image warps build the first left/right grids.
        outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        for d in deltas:
            # Next-stage feature warps use exactly the same flow version and should
            # reuse the two grids built by the prior image warps.
            outs += [net.warp(f0, flow[:, :2]), net.warp(f1, flow[:, 2:4])]
            flow.add_(d)
            outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        return outs

    expected = sequence()
    opt = mod._IFNetWarpGridCache(net)
    try:
        opt.begin_forward()
        actual = sequence()
        opt.end_forward()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        snap = opt.snapshot()
        assert snap["builds"] == 10
        assert snap["reuses"] == 8
        assert snap["peak_entries"] == 2
        assert snap["forwards"] == 1
        assert snap["fallbacks"] == 0
    finally:
        opt.close()
    assert "warp" not in net.__dict__

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert req["warp_input_cache"][0] == ["off", "fp32_reuse", "grid_reuse"]
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"
    assert "warp_grid_cache" not in req


def test_dev2112_grid_reuse_works_with_inference_tensors_via_block_hooks():
    class FakeBlock(torch.nn.Module):
        def forward(self, x):
            return x

    class FakeWarpNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._warp_grids = {}
            self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(5)])
        def build(self, H, W, device):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

    net = FakeWarpNet()
    net.build(8, 10, torch.device("cpu"))
    img0 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    img1 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    f0 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    f1 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    base_flow = torch.randn(1, 4, 8, 10, dtype=torch.float32) * 0.2
    deltas = [torch.randn_like(base_flow) * 0.03 for _ in range(4)]

    def sequence():
        flow = base_flow.clone()
        outs = []
        # block0 completes before the first image warps.
        _ = net.blocks[0](flow)
        outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        for i, d in enumerate(deltas, start=1):
            # These feature warps reuse the prior image grids. The block hook then
            # invalidates them before flow.add_ changes the field for image warps.
            outs += [net.warp(f0, flow[:, :2]), net.warp(f1, flow[:, 2:4])]
            _ = net.blocks[i](flow)
            flow.add_(d)
            outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        return outs

    with torch.inference_mode():
        expected = sequence()
        # Confirm why DEV2.11 failed on the real workstation: inference tensors do
        # not provide a version counter that can be used as a cache key.
        probe = base_flow.clone()
        try:
            _ = probe._version
            version_unavailable = False
        except RuntimeError:
            version_unavailable = True
        assert version_unavailable

        opt = mod._IFNetWarpGridCache(net)
        try:
            assert opt.strategy == "block_hooks"
            opt.begin_forward()
            actual = sequence()
            opt.end_forward()
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            snap = opt.snapshot()
            assert snap["builds"] == 10
            assert snap["reuses"] == 8
            assert snap["peak_entries"] == 2
            assert snap["forwards"] == 1
            assert snap["fallbacks"] == 0
            assert snap["invalidations"] == 5
            assert snap["strategy"] == "block_hooks"
        finally:
            opt.close()
    assert "warp" not in net.__dict__


def test_dev212_block4_detail_profiler_preserves_output_and_profiles_children():
    import types, sys
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = lambda *a, **k: None
    comfy.model_management = mm
    old_comfy = sys.modules.get("comfy")
    old_mm = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

    class ToyRes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Identity()
            self.relu = torch.nn.Identity()
        def forward(self, x):
            # Keep exact output while exercising the same child-module order as
            # ComfyUI ResConv: conv -> residual arithmetic -> relu.
            y = self.conv(x)
            y = x + y * 0.0
            return self.relu(y)

    class ToyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
            self.convblock = torch.nn.Sequential(*[ToyRes() for _ in range(8)])
            self.lastconv = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        def forward(self, x):
            x = self.conv0(x)
            x = self.convblock(x)
            return self.lastconv(x)

    class ToyIFNet(torch.nn.Module):
        pad_align = 1
        def __init__(self):
            super().__init__()
            self.encode = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(5)])
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        def memory_used_forward(self, shape, dtype):
            return 0
        def forward(self, a, b, timestep, cache=None):
            self.encode(a); self.encode(b)
            x = (a * (1 - timestep) + b * timestep) * self.scale
            for block in self.blocks:
                x = block(x)
            return x

    class Patcher:
        load_device = torch.device("cpu")
        _longvideo_model_name = "rife_v4.25.safetensors"
        def __init__(self): self.model = ToyIFNet()
        def model_dtype(self): return torch.float32

    try:
        images = torch.rand(4, 8, 8, 3)
        s0, s1 = {}, {}
        out0 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            Patcher(), images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="off", feature_cache="off", warp_grid_cache="off", stats=s0,
        )), dim=0)
        out1 = torch.cat(list(mod.iter_interpolated_sequence_chunks(
            Patcher(), images, 2, scene_cut=False, cpu_output="float32",
            pair_batch=1, chunk_frames=64, d2h_mode="sync", h2d_mode="sync",
            ifnet_profile="block4_detail", feature_cache="off", warp_grid_cache="off", stats=s1,
        )), dim=0)
        assert torch.equal(out0, out1)
        assert s1["ifnet_profile_active"] == "block4_detail"
        assert s1["ifnet_stage_calls"]["block4"] == 3
        assert s1["ifnet_stage_calls"]["block4_conv0_0"] == 3
        assert s1["ifnet_stage_calls"]["block4_conv0_1"] == 3
        assert s1["ifnet_stage_calls"]["block4_deconv"] == 3
        assert s1["ifnet_stage_calls"]["block4_pixelshuffle"] == 3
        for i in range(8):
            assert s1["ifnet_stage_calls"][f"block4_res{i}"] == 3
            assert s1["ifnet_stage_calls"][f"block4_res{i}_conv"] == 3
            assert s1["ifnet_stage_calls"][f"block4_res{i}_relu"] == 3
        # Nested child timings must not corrupt the parent IFNet attribution.
        assert s1["ifnet_stage_unattributed_seconds"] >= 0.0
    finally:
        if old_comfy is None: sys.modules.pop("comfy", None)
        else: sys.modules["comfy"] = old_comfy
        if old_mm is None: sys.modules.pop("comfy.model_management", None)
        else: sys.modules["comfy.model_management"] = old_mm


def test_dev214_block4_resconv_inplace_is_bit_exact_and_restores_forwards():
    class ToyRes(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = torch.nn.Conv2d(c, c, 3, 1, 1)
            self.beta = torch.nn.Parameter(torch.randn(1, c, 1, 1))
            self.relu = torch.nn.LeakyReLU(0.2, True)
        def forward(self, x):
            return self.relu(torch.addcmul(x, self.conv(x), self.beta))

    class ToyBlock(torch.nn.Module):
        def __init__(self, c=8):
            super().__init__()
            self.convblock = torch.nn.Sequential(*[ToyRes(c) for _ in range(8)])

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)] + [ToyBlock()])

    torch.manual_seed(214)
    net = ToyNet().eval()
    x = torch.randn(1, 8, 16, 16)
    with torch.inference_mode():
        expected = net.blocks[4].convblock(x.clone())
        opt = mod._IFNetBlock4ResConvInplace(net)
        try:
            actual = net.blocks[4].convblock(x.clone())
            assert torch.equal(actual, expected)
            snap = opt.snapshot()
            assert snap["patched"] == 8
            assert snap["calls"] == 8
        finally:
            opt.close()
        restored = net.blocks[4].convblock(x.clone())
        assert torch.equal(restored, expected)
        assert all("forward" not in r.__dict__ for r in net.blocks[4].convblock)

    # The existing serialized slot is reused; no new widget is introduced.
    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert req["ifnet_profile"][0] == ["off", "stages", "detail", "warp_detail", "block4_detail", "block4_res_inplace", "block4_scale1_bypass", "conv_family_detail", "outer_detail", "outer_cat_pushdown", "warp_postopt_detail"]
    assert req["feature_cache"][1]["default"] == "adjacent"
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"


def test_dev215_block4_scale1_bypass_is_bit_exact_and_restores_forward():
    class ToyRes(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = torch.nn.Conv2d(c, c, 3, 1, 1)
            self.beta = torch.nn.Parameter(torch.randn(1, c, 1, 1))
            self.relu = torch.nn.LeakyReLU(0.2, True)
        def forward(self, x):
            return self.relu(torch.addcmul(x, self.conv(x), self.beta))

    class ToyBlock(torch.nn.Module):
        def __init__(self, in_ch=12, c=8):
            super().__init__()
            self.conv0 = torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Conv2d(in_ch + 4, c // 2, 3, 2, 1), torch.nn.LeakyReLU(0.2, True)),
                torch.nn.Sequential(torch.nn.Conv2d(c // 2, c, 3, 2, 1), torch.nn.LeakyReLU(0.2, True)),
            )
            self.convblock = torch.nn.Sequential(*[ToyRes(c) for _ in range(8)])
            self.lastconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), torch.nn.PixelShuffle(2))
        def forward(self, x, flow=None, scale=1):
            x = torch.nn.functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
            if flow is not None:
                flow = torch.nn.functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
                x = torch.cat((x, flow), 1)
            feat = self.convblock(self.conv0(x))
            tmp = torch.nn.functional.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)] + [ToyBlock()])

    torch.manual_seed(215)
    net = ToyNet().eval()
    x = torch.randn(1, 12, 32, 32)
    flow = torch.randn(1, 4, 32, 32)
    block4 = net.blocks[4]
    with torch.inference_mode():
        expected = tuple(t.clone() for t in block4(x.clone(), flow.clone(), scale=1))
        opt = mod._IFNetBlock4Scale1Bypass(net)
        try:
            actual = block4(x.clone(), flow.clone(), scale=1)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            snap = opt.snapshot()
            assert snap["calls"] == 1
            assert snap["bypassed_interpolates"] == 3
            assert snap["fallbacks"] == 0
            # Any unexpected non-unit scale must use the untouched stock path.
            stock_half = tuple(t.clone() for t in opt._original_forward(x.clone(), flow.clone(), scale=2))
            patched_half = block4(x.clone(), flow.clone(), scale=2)
            assert all(torch.equal(a, b) for a, b in zip(patched_half, stock_half))
            assert opt.snapshot()["fallbacks"] == 1
        finally:
            opt.close()
        restored = block4(x.clone(), flow.clone(), scale=1)
        assert all(torch.equal(a, b) for a, b in zip(restored, expected))
        assert "forward" not in block4.__dict__

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert "block4_scale1_bypass" in req["ifnet_profile"][0]
    assert req["feature_cache"][1]["default"] == "adjacent"
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"


def test_dev216_conv_family_profiler_is_output_neutral_and_profiles_all_blocks():
    class ToyRes(torch.nn.Module):
        def __init__(self, c=4):
            super().__init__()
            self.conv = torch.nn.Conv2d(c, c, 3, 1, 1)
            self.beta = torch.nn.Parameter(torch.ones(1, c, 1, 1))
            self.relu = torch.nn.LeakyReLU(0.2, True)
        def forward(self, x):
            return self.relu(torch.addcmul(x, self.conv(x), self.beta))

    class ToyBlock(torch.nn.Module):
        def __init__(self, c=4):
            super().__init__()
            self.conv0 = torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Conv2d(c, c, 3, 1, 1), torch.nn.LeakyReLU(0.2, True)),
                torch.nn.Sequential(torch.nn.Conv2d(c, c, 3, 1, 1), torch.nn.LeakyReLU(0.2, True)),
            )
            self.convblock = torch.nn.Sequential(*[ToyRes(c) for _ in range(8)])
            self.lastconv = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        def forward(self, x):
            return self.lastconv(self.convblock(self.conv0(x)))

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(5)])
        def forward(self, x):
            self.encode(x)
            for block in self.blocks:
                x = block(x)
            return x

    torch.manual_seed(216)
    net = ToyNet().eval()
    x = torch.randn(1, 4, 8, 8)
    with torch.inference_mode():
        expected = net(x.clone())
        prof = mod._IFNetStageProfiler(net, profile_conv_family=True)
        try:
            prof.begin("cpu")
            actual = net(x.clone())
            trace = prof.end()
            assert torch.equal(actual, expected)
            events = trace["events"]
            for bi in range(5):
                assert len(events[f"block{bi}_conv0_0"]) == 1
                assert len(events[f"block{bi}_conv0_1"]) == 1
                assert len(events[f"block{bi}_deconv"]) == 1
                assert len(events[f"block{bi}_pixelshuffle"]) == 1
                for ri in range(8):
                    assert len(events[f"block{bi}_res{ri}_conv"]) == 1
        finally:
            prof.close()
        restored = net(x.clone())
        assert torch.equal(restored, expected)

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert "conv_family_detail" in req["ifnet_profile"][0]
    assert req["feature_cache"][1]["default"] == "adjacent"
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"



def test_dev217_outer_profiler_is_bit_exact_and_counts_current_ifnet_outer_ops():
    class ToyBlock(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)
        def forward(self, x, flow=None, scale=1):
            base = x[:, :1] * 0.0 + self.value
            fd = base.repeat(1, 4, 1, 1)
            mask = base
            feat = base.repeat(1, 2, 1, 1)
            return fd, mask, feat

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = torch.nn.Conv2d(3, 2, 1, bias=False)
            self.blocks = torch.nn.ModuleList([ToyBlock(i + 1) for i in range(5)])
            self.scale_list = [16, 8, 4, 2, 1]
            self._warp_grids = {}
        def _build_warp_grids(self, H, W, device):
            self._warp_grids[(H, W)] = True
        def warp(self, img, flow):
            return img + flow[:, :1] * 0.0
        def forward(self, img0, img1, timestep=0.5, cache=None):
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, device=img0.device, dtype=img0.dtype)
            self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
            B = img0.shape[0]
            f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else self.encode(img0)
            f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else self.encode(img1)
            flow = mask = feat = None
            warped_img0, warped_img1 = img0, img1
            for i, block in enumerate(self.blocks):
                if flow is None:
                    flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[i])
                else:
                    fd, mask, feat = block(
                        torch.cat((warped_img0, warped_img1, self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4]), timestep, mask, feat), 1),
                        flow, scale=self.scale_list[i])
                    flow = flow.add_(fd)
                warped_img0 = self.warp(img0, flow[:, :2])
                warped_img1 = self.warp(img1, flow[:, 2:4])
            return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))

    torch.manual_seed(217)
    net = ToyNet().eval()
    img0 = torch.randn(1, 3, 8, 8)
    img1 = torch.randn(1, 3, 8, 8)
    t = torch.full((1, 1, 8, 8), 0.5)
    with torch.inference_mode():
        expected = net(img0.clone(), img1.clone(), timestep=t.clone())
        prof = mod._IFNetStageProfiler(net, profile_outer=True)
        try:
            assert prof.usable
            prof.begin("cpu")
            actual = net(img0.clone(), img1.clone(), timestep=t.clone())
            trace = prof.end()
            assert torch.equal(actual, expected)
            events = trace["events"]
            assert len(events["outer_cat_initial"]) == 1
            assert len(events["outer_cat_refine"]) == 4
            assert len(events["outer_warp_feature"]) == 8
            assert len(events["outer_flow_add"]) == 4
            assert len(events["outer_warp_image"]) == 10
            assert len(events["outer_sigmoid"]) == 1
            assert len(events["outer_lerp"]) == 1
        finally:
            prof.close()
        restored = net(img0.clone(), img1.clone(), timestep=t.clone())
        assert torch.equal(restored, expected)
        assert "forward" not in net.__dict__

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert "outer_detail" in req["ifnet_profile"][0]
    assert req["feature_cache"][1]["default"] == "adjacent"
    assert req["warp_input_cache"][1]["default"] == "grid_reuse"


def test_dev218_outer_cat_pushdown_fp16_exact_and_restores():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = nn.Conv2d(c, c, 3, 1, 1).half()
            self.beta = nn.Parameter(torch.ones(1, c, 1, 1, dtype=torch.float16))
            self.relu = nn.LeakyReLU(0.2, True)
        def forward(self, x):
            return self.relu(torch.addcmul(x, self.conv(x), self.beta))

    class Block(nn.Module):
        def __init__(self, in_planes, c=8):
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Sequential(nn.Conv2d(in_planes, c // 2, 3, 2, 1).half(), nn.LeakyReLU(0.2, True)),
                nn.Sequential(nn.Conv2d(c // 2, c, 3, 2, 1).half(), nn.LeakyReLU(0.2, True)),
            )
            self.convblock = nn.Sequential(*[Res(c) for _ in range(2)])
            self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1).half(), nn.PixelShuffle(2))
        def forward(self, x, flow=None, scale=1):
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
                x = torch.cat((x, flow), 1)
            feat = self.convblock(self.conv0(x))
            tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = nn.Conv2d(3, 4, 1).half()
            self.blocks = nn.ModuleList([Block(15), Block(28), Block(28), Block(28), Block(28)])
            self.scale_list = [16, 8, 4, 2, 1]
        def get_dtype(self): return torch.float16
        def _build_warp_grids(self, H, W, device): return None
        def warp(self, img, flow): return img
        def extract_features(self, img): return self.encode(img)
        def forward(self, img0, img1, timestep=0.5, cache=None):
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, device=img0.device, dtype=img0.dtype)
            B = img0.shape[0]
            f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else self.encode(img0)
            f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else self.encode(img1)
            flow = mask = feat = None
            warped_img0, warped_img1 = img0, img1
            for i, block in enumerate(self.blocks):
                if flow is None:
                    flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[i])
                else:
                    fd, mask, feat = block(
                        torch.cat((warped_img0, warped_img1, self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4]), timestep, mask, feat), 1),
                        flow, scale=self.scale_list[i])
                    flow = flow.add_(fd)
                warped_img0 = self.warp(img0, flow[:, :2])
                warped_img1 = self.warp(img1, flow[:, 2:4])
            return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))

    torch.manual_seed(218)
    net = ToyNet().eval()
    img0 = torch.randn(1, 3, 64, 64).half()
    img1 = torch.randn(1, 3, 64, 64).half()
    with torch.inference_mode():
        expected = net(img0.clone(), img1.clone())
        hook_calls = [0, 0, 0, 0]
        handles = []
        for bi in range(4):
            def _hook(_m, _i, _o, _bi=bi):
                hook_calls[_bi] += 1
            handles.append(net.blocks[bi].register_forward_hook(_hook))
        scale_opt = mod._IFNetBlock4Scale1Bypass(net)
        cat_opt = mod._IFNetOuterCatPushdown(net)
        try:
            actual = net(img0.clone(), img1.clone())
            assert torch.equal(actual, expected)
            snap = cat_opt.snapshot()
            assert snap == {"forwards": 1, "pushed_cats": 4, "component_interpolates": 26, "fallbacks": 0}
            ss = scale_opt.snapshot()
            assert ss["calls"] == 1 and ss["bypassed_interpolates"] == 3 and ss["fallbacks"] == 0
            assert hook_calls == [1, 1, 1, 1]
        finally:
            cat_opt.close()
            scale_opt.close()
            for h in handles:
                h.remove()
        restored = net(img0.clone(), img1.clone())
        assert torch.equal(restored, expected)
        assert "forward" not in net.__dict__
        assert "forward" not in net.blocks[4].__dict__

    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert "outer_cat_pushdown" in req["ifnet_profile"][0]



def test_dev219_postopt_warp_profiler_preserves_grid_reuse_and_exact_output():
    class FakeBlock(torch.nn.Module):
        def forward(self, x):
            return x

    class FakeWarpNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._warp_grids = {}
            self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(5)])
        def build(self, H, W, device):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij"
            )
            self._warp_grids[(H, W)] = (
                torch.stack((gx, gy), dim=0).unsqueeze(0),
                torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device),
            )
        def warp(self, img, flow):
            B, _, H, W = img.shape
            base_grid, flow_div = self._warp_grids[(H, W)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

    net = FakeWarpNet()
    net.build(8, 10, torch.device("cpu"))
    img0 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    img1 = torch.rand(1, 3, 8, 10, dtype=torch.float64)
    f0 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    f1 = torch.rand(1, 4, 8, 10, dtype=torch.float64)
    base_flow = torch.randn(1, 4, 8, 10, dtype=torch.float32) * 0.2
    deltas = [torch.randn_like(base_flow) * 0.03 for _ in range(4)]

    def sequence():
        flow = base_flow.clone()
        outs = []
        _ = net.blocks[0](flow)
        outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        for i, d in enumerate(deltas, start=1):
            outs += [net.warp(f0, flow[:, :2]), net.warp(f1, flow[:, 2:4])]
            _ = net.blocks[i](flow)
            flow.add_(d)
            outs += [net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])]
        return outs

    with torch.inference_mode():
        expected = sequence()
        opt = mod._IFNetWarpGridCache(net, profile_inner=True)
        try:
            opt.begin_forward("cpu")
            actual = sequence()
            trace = opt.end_forward()
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            snap = opt.snapshot()
            assert snap["builds"] == 10
            assert snap["reuses"] == 8
            assert snap["fallbacks"] == 0
            events = trace["events"]
            assert len(events["warp_post_image_norm"]) == 10
            assert len(events["warp_post_image_grid"]) == 10
            assert len(events["warp_post_image_cast_in"]) == 10
            assert len(events["warp_post_image_sample"]) == 10
            assert len(events["warp_post_image_cast_out"]) == 10
            assert "warp_post_feature_norm" not in events
            assert "warp_post_feature_grid" not in events
            assert len(events["warp_post_feature_cast_in"]) == 8
            assert len(events["warp_post_feature_sample"]) == 8
            assert len(events["warp_post_feature_cast_out"]) == 8
        finally:
            opt.close()
    assert "warp" not in net.__dict__
    req = mod.LV_LongVideoFrameInterpolationStream.INPUT_TYPES()["required"]
    assert "warp_postopt_detail" in req["ifnet_profile"][0]
