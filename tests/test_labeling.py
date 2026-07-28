from labeling import (
    AnnotationStore,
    LabelingSession,
    UndoHistory,
    ViewTransform,
    color_for,
    hit_test,
    move_box,
    new_box_from_drag,
    resize_box,
    resize_mode_at,
)

SAMPLE = (
    "1,7,100.0,50.0,40.0,60.0,1,-1,-1,-1\n"
    "1,9,300.0,80.0,40.0,60.0,1,-1,-1,-1\n"
    "2,7,104.0,52.0,40.0,60.0,1,-1,-1,-1\n"
)


def test_store_roundtrip():
    store = AnnotationStore.from_text(SAMPLE)
    assert len(store.boxes_at(1)) == 2
    assert store.all_ids() == {7, 9}
    reparsed = AnnotationStore.from_text(store.to_text())
    assert reparsed.to_text() == store.to_text()


def test_store_next_id():
    store = AnnotationStore.from_text(SAMPLE)
    assert store.next_id() == 10


def test_store_delete_and_rename():
    store = AnnotationStore.from_text(SAMPLE)
    assert store.delete_by_id(9) == 1
    assert 9 not in store.all_ids()
    assert store.rename_id(7, 42) == 2
    assert store.all_ids() == {42}


def test_store_propagate():
    store = AnnotationStore.from_text(SAMPLE)
    added = store.propagate(7, start_frame=1, final_frame=4)
    assert added == 2
    assert any(b["track_id"] == 7 for b in store.boxes_at(3))


def test_snapshot_restore():
    store = AnnotationStore.from_text(SAMPLE)
    snap = store.snapshot()
    store.delete_by_id(7)
    store.restore(snap)
    assert store.all_ids() == {7, 9}


def test_view_transform_roundtrip():
    view = ViewTransform(1000, 500)
    view.fit(2000, 1000)
    assert view.scale == 0.5
    box = {"track_id": 1, "x": 100, "y": 40, "width": 20, "height": 30}
    x, y, w, h = view.to_canvas(box)
    assert (w, h) == (10, 15)


def test_hit_test_prefers_smaller():
    view = ViewTransform(1000, 1000)
    view.fit(1000, 1000)
    boxes = [
        {"track_id": 1, "x": 0, "y": 0, "width": 500, "height": 500},
        {"track_id": 2, "x": 100, "y": 100, "width": 50, "height": 50},
    ]
    assert hit_test(boxes, view, 120, 120) == 1


def test_resize_mode_corner():
    assert resize_mode_at((10, 10, 100, 100), 108, 108) == "resize_br"


def test_move_and_resize_clamp():
    box = {"track_id": 1, "x": 10, "y": 10, "width": 40, "height": 40}
    move_box(box, -100, -100, 640, 480)
    assert box["x"] == 0 and box["y"] == 0
    resize_box(box, "resize_br", 10, 10, 640, 480)
    assert box["width"] == 50 and box["height"] == 50


def test_new_box_from_drag_min_side():
    view = ViewTransform(1000, 1000)
    view.fit(1000, 1000)
    assert new_box_from_drag(view, (10, 10), (12, 12), track_id=5) is None
    box = new_box_from_drag(view, (10, 10), (60, 80), track_id=5)
    assert box["track_id"] == 5


def test_history_capacity():
    history = UndoHistory(capacity=2)
    for value in (1, 2, 3):
        history.push(value)
    assert history.pop() == 3
    assert history.pop() == 2
    assert not history.can_undo()


def test_color_stable():
    assert color_for(0) == color_for(12)


def test_session_edits(tmp_path):
    seq = tmp_path / "V001"
    (seq / "gt").mkdir(parents=True)
    (seq / "img1").mkdir()
    (seq / "gt" / "gt.txt").write_text(SAMPLE)
    for name in ("000001.jpg", "000002.jpg"):
        (seq / "img1" / name).write_bytes(b"x")
    session = LabelingSession(str(seq), auto_save=True)
    assert session.total_frames == 2
    session.rename_id(7, 70)
    assert 70 in session.store.all_ids()
    session.undo()
    assert 7 in session.store.all_ids()
    reloaded = AnnotationStore.from_file(str(seq / "gt" / "gt.txt"))
    assert 7 in reloaded.all_ids()
