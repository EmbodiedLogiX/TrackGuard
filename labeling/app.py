from __future__ import annotations

import os
from typing import Optional

from labeling.geometry import (
    ViewTransform,
    hit_test,
    move_box,
    new_box_from_drag,
    resize_box,
    resize_mode_at,
)
from labeling.palette import color_for
from labeling.session import LabelingSession


def _toolkit():
    import tkinter as tk
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import BOTH, LEFT, RIGHT, X, Y, YES
    from PIL import Image, ImageTk
    return tk, ttk, Image, ImageTk, dict(BOTH=BOTH, LEFT=LEFT, RIGHT=RIGHT,
                                         X=X, Y=Y, YES=YES)


class ReviewApp:
    def __init__(self, directory: Optional[str] = None, auto_save: bool = True,
                 canvas_width: int = 1100, canvas_height: int = 700,
                 themename: str = "cosmo"):
        tk, ttk, Image, ImageTk, const = _toolkit()
        self._tk = tk
        self._ttk = ttk
        self._Image = Image
        self._ImageTk = ImageTk
        self._c = const

        self.window = ttk.Window(themename=themename)
        self.window.title("TrackGuard Annotation Review")
        self.window.geometry("1400x900")

        self.view = ViewTransform(canvas_width, canvas_height)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.auto_save = auto_save
        self.session: Optional[LabelingSession] = None
        self.selected: Optional[int] = None
        self.photo = None
        self.image_cache = {}
        self.cache_size = 5
        self.create_mode = False
        self.create_start = None
        self.create_preview = None
        self.drag = {"index": None, "x": 0, "y": 0, "mode": None}

        self._build_ui()
        self._bind_keys()
        if directory:
            self.open(directory)

    def _build_ui(self):
        tk, ttk, c = self._tk, self._ttk, self._c
        self.status = ttk.Label(self.window, text="Open a sequence (gt/ + img1/)",
                                bootstyle="inverse-primary", padding=5)
        self.status.pack(fill=c["X"])
        main = ttk.Frame(self.window, padding=10)
        main.pack(fill=c["BOTH"], expand=c["YES"])

        left = ttk.Frame(main, width=200)
        left.pack(side=c["LEFT"], fill=c["Y"], padx=5)
        ttk.Label(left, text="Frames", font=("Arial", 12, "bold")).pack(fill=c["X"], pady=5)
        listbox_frame = ttk.Frame(left)
        listbox_frame.pack(fill=c["BOTH"], expand=c["YES"])
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=c["RIGHT"], fill=c["Y"])
        self.frame_list = tk.Listbox(listbox_frame, width=28, font=("Consolas", 10),
                                     yscrollcommand=scrollbar.set)
        self.frame_list.pack(side=c["LEFT"], fill=c["BOTH"], expand=c["YES"])
        scrollbar.config(command=self.frame_list.yview)
        self.frame_list.bind("<<ListboxSelect>>", self._on_pick_frame)
        ttk.Button(left, text="Open sequence", command=self._prompt_open,
                   bootstyle="info").pack(pady=8, fill=c["X"])

        center = ttk.Frame(main)
        center.pack(side=c["LEFT"], fill=c["BOTH"], expand=c["YES"], padx=10)
        self.canvas = tk.Canvas(center, width=self.canvas_width, height=self.canvas_height,
                                bg="#f0f0f0", bd=2, relief="solid")
        self.canvas.pack(pady=10)
        nav = ttk.Frame(center)
        nav.pack(fill=c["X"], pady=6)
        ttk.Button(nav, text="<< Prev (A)", command=self.prev_frame,
                   bootstyle="outline").pack(side=c["LEFT"], padx=5)
        self.frame_label = ttk.Label(nav, text="Frame 0 / 0", font=("Arial", 11, "bold"))
        self.frame_label.pack(side=c["LEFT"], padx=15)
        ttk.Button(nav, text="Next (D) >>", command=self.next_frame,
                   bootstyle="outline").pack(side=c["LEFT"], padx=5)
        ttk.Label(nav, text="From ID:").pack(side=c["LEFT"], padx=6)
        self.src_entry = ttk.Entry(nav, width=7)
        self.src_entry.pack(side=c["LEFT"])
        ttk.Label(nav, text="To ID:").pack(side=c["LEFT"], padx=6)
        self.dst_entry = ttk.Entry(nav, width=7)
        self.dst_entry.pack(side=c["LEFT"])
        ttk.Button(nav, text="Rename all", command=self._rename_all,
                   bootstyle="warning").pack(side=c["LEFT"], padx=8)

        bulk = ttk.Frame(center)
        bulk.pack(fill=c["X"], pady=6)
        ttk.Label(bulk, text="Propagate ID:").pack(side=c["LEFT"], padx=5)
        self.bulk_id_entry = ttk.Entry(bulk, width=7)
        self.bulk_id_entry.pack(side=c["LEFT"], padx=5)
        ttk.Label(bulk, text="until frame:").pack(side=c["LEFT"], padx=5)
        self.bulk_final_entry = ttk.Entry(bulk, width=7)
        self.bulk_final_entry.pack(side=c["LEFT"], padx=5)
        ttk.Button(bulk, text="Propagate", command=self._propagate,
                   bootstyle="success").pack(side=c["LEFT"], padx=8)

        right = ttk.Labelframe(main, text="Selection", padding=15)
        right.pack(side=c["RIGHT"], fill=c["Y"], padx=12)
        self.selected_label = ttk.Label(right, text="No box selected",
                                        font=("Arial", 13, "bold"), bootstyle="primary")
        self.selected_label.pack(fill=c["X"], pady=8)
        self.detail_label = ttk.Label(right, text="", font=("Arial", 10))
        self.detail_label.pack(anchor="w", pady=4)
        id_row = ttk.Frame(right)
        id_row.pack(fill=c["X"], pady=6)
        ttk.Label(id_row, text="Track ID:", font=("Arial", 11, "bold")).pack(side=c["LEFT"])
        self.id_var = tk.StringVar()
        ttk.Entry(id_row, textvariable=self.id_var, width=10).pack(side=c["LEFT"], padx=5)
        ttk.Button(id_row, text="Update", command=self._update_id,
                   bootstyle="outline").pack(side=c["LEFT"], padx=3)
        ttk.Button(right, text="Delete this ID everywhere", command=self._delete_id,
                   bootstyle="danger").pack(fill=c["X"], pady=4)

        create = ttk.Labelframe(right, text="Create", padding=10)
        create.pack(fill=c["X"], pady=8)
        row = ttk.Frame(create)
        row.pack(fill=c["X"])
        ttk.Label(row, text="New ID:").pack(side=c["LEFT"], padx=4)
        self.new_id_var = tk.StringVar(value="100")
        ttk.Entry(row, textvariable=self.new_id_var, width=8).pack(side=c["LEFT"], padx=4)
        self.create_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(create, text="Enable drag-to-create", variable=self.create_var,
                        command=self._toggle_create, bootstyle="info").pack(fill=c["X"], pady=4)

        ttk.Button(right, text="Delete selected (W)", command=self._delete_selected,
                   bootstyle="danger").pack(fill=c["X"], pady=4)
        actions = ttk.Frame(right)
        actions.pack(fill=c["X"], pady=8)
        ttk.Button(actions, text="Save", command=self._save,
                   bootstyle="success").pack(side=c["LEFT"], padx=2)
        ttk.Button(actions, text="Reset", command=self._reset,
                   bootstyle="secondary").pack(side=c["LEFT"], padx=2)
        tips = ("A/Left prev  D/Right next\nspace toggle boxes\nQ undo  E quick-create\n"
                "W/Del delete  click select  drag move/resize")
        ttk.Label(right, text=tips, font=("Arial", 9)).pack(fill=c["X"], pady=8)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _bind_keys(self):
        for key in ("<Left>", "<a>", "<A>"):
            self.window.bind(key, lambda e: self.prev_frame())
        for key in ("<Right>", "<d>", "<D>"):
            self.window.bind(key, lambda e: self.next_frame())
        for key in ("<Delete>", "<BackSpace>", "<w>", "<W>"):
            self.window.bind(key, lambda e: self._delete_selected())
        for key in ("<q>", "<Q>"):
            self.window.bind(key, lambda e: self._undo())
        for key in ("<e>", "<E>"):
            self.window.bind(key, lambda e: self._quick_create())
        self.window.bind("<space>", lambda e: self._render())

    def _prompt_open(self):
        from tkinter import filedialog
        directory = filedialog.askdirectory(title="Select sequence (gt/ + img1/)")
        if directory:
            self.open(directory)

    def open(self, directory: str):
        from tkinter import messagebox
        try:
            self.session = LabelingSession(directory, auto_save=self.auto_save)
        except FileNotFoundError as exc:
            messagebox.showerror("Error", f"Missing gt/gt.txt or img1/: {exc}")
            return
        if not self.session.images:
            messagebox.showwarning("Warning", "No images in img1/")
            return
        self.selected = None
        self.image_cache.clear()
        self._fill_frame_list()
        self.status.config(text=f"Loaded {self.session.name}")
        self._render()

    def _fill_frame_list(self):
        self.frame_list.delete(0, self._tk.END)
        for i, name in enumerate(self.session.images):
            self.frame_list.insert(self._tk.END, f"{i + 1}: {name}")

    def _load_photo(self, path: str):
        if path in self.image_cache:
            return self.image_cache[path]
        if len(self.image_cache) >= self.cache_size:
            self.image_cache.pop(next(iter(self.image_cache)))
        image = self._Image.open(path)
        self.view.fit(*image.size)
        new_size = (int(image.width * self.view.scale), int(image.height * self.view.scale))
        resized = image.resize(new_size, self._Image.Resampling.LANCZOS)
        photo = self._ImageTk.PhotoImage(resized)
        self.image_cache[path] = photo
        return photo

    def _image_size(self, path: str):
        with self._Image.open(path) as image:
            return image.size

    def _render(self):
        if self.session is None:
            return
        path = self.session.image_path()
        self.photo = self._load_photo(path)
        self.view.fit(*self._image_size(path))
        self.canvas.delete("all")
        self.canvas.create_image(self.view.offset_x, self.view.offset_y,
                                 image=self.photo, anchor="nw")
        for index, box in enumerate(self.session.current_boxes()):
            x, y, w, h = self.view.to_canvas(box)
            color = color_for(box["track_id"])
            width = 6 if index == self.selected else 4
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=width)
            self.canvas.create_text(x + 4, y - 4, text=f"ID:{box['track_id']}",
                                    fill=color, font=("Arial", 14, "bold"), anchor="sw")
        total = self.session.total_frames
        self.frame_label.config(text=f"Frame {self.session.frame_number} / {total}")
        self.frame_list.selection_clear(0, self._tk.END)
        self.frame_list.selection_set(self.session.frame_index)
        self.frame_list.see(self.session.frame_index)
        self._refresh_detail()

    def _refresh_detail(self):
        boxes = self.session.current_boxes() if self.session else []
        if self.selected is not None and self.selected < len(boxes):
            box = boxes[self.selected]
            self.selected_label.config(text=f"Box #{self.selected + 1}", bootstyle="success")
            self.detail_label.config(
                text=f"pos ({box['x']:.1f}, {box['y']:.1f})  "
                     f"size {box['width']:.1f} x {box['height']:.1f}")
            self.id_var.set(str(box["track_id"]))
        else:
            self.selected_label.config(text="No box selected", bootstyle="primary")
            self.detail_label.config(text="")

    def _on_pick_frame(self, event):
        selection = self.frame_list.curselection()
        if selection and self.session:
            self.session.go_to(selection[0])
            self.selected = None
            self._render()

    def next_frame(self):
        if self.session:
            self.session.next_frame()
            self.selected = None
            self._render()

    def prev_frame(self):
        if self.session:
            self.session.prev_frame()
            self.selected = None
            self._render()

    def _toggle_create(self):
        self.create_mode = self.create_var.get()
        self.create_start = None

    def _quick_create(self):
        if self.session:
            self.new_id_var.set(str(self.session.next_id()))
        self.create_var.set(True)
        self._toggle_create()

    def _on_click(self, event):
        if self.session is None:
            return
        if self.create_mode:
            self.create_start = (event.x, event.y)
            return
        boxes = self.session.current_boxes()
        index = hit_test(boxes, self.view, event.x, event.y)
        self.selected = index
        if index is not None:
            self.drag = {"index": index, "x": event.x, "y": event.y,
                         "mode": resize_mode_at(self.view.to_canvas(boxes[index]),
                                                event.x, event.y)}
        self._refresh_detail()
        self._render()

    def _on_move(self, event):
        if self.session is None:
            return
        if self.create_mode and self.create_start:
            if self.create_preview:
                self.canvas.delete(self.create_preview)
            x1, y1 = self.create_start
            self.create_preview = self.canvas.create_rectangle(
                x1, y1, event.x, event.y, outline="#00FF00", width=2, dash=(5, 5))
            return
        if self.selected is None or self.drag["index"] is None:
            return
        boxes = self.session.current_boxes()
        if self.selected >= len(boxes):
            return
        dx_img, dy_img = self.view.delta_to_image(event.x - self.drag["x"],
                                                  event.y - self.drag["y"])
        width, height = self._image_size(self.session.image_path())
        box = boxes[self.selected]
        if self.drag["mode"]:
            resize_box(box, self.drag["mode"], dx_img, dy_img, width, height)
        else:
            move_box(box, dx_img, dy_img, width, height)
        self.drag["x"], self.drag["y"] = event.x, event.y
        self._render()

    def _on_release(self, event):
        if self.session is None:
            return
        if self.create_mode and self.create_start:
            width, height = self._image_size(self.session.image_path())
            try:
                track_id = int(self.new_id_var.get())
            except ValueError:
                track_id = 100
            box = new_box_from_drag(self.view, self.create_start, (event.x, event.y), track_id)
            if box is not None:
                box["x"] = max(0.0, min(box["x"], width - box["width"]))
                box["y"] = max(0.0, min(box["y"], height - box["height"]))
                self.session.add_box(box)
            self.create_start = None
            if self.create_preview:
                self.canvas.delete(self.create_preview)
                self.create_preview = None
            self._render()
            return
        if self.selected is not None:
            self.session._persist()
        self.drag["index"] = None

    def _update_id(self):
        from tkinter import messagebox
        if self.session is None or self.selected is None:
            return
        try:
            track_id = int(self.id_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter a numeric ID")
            return
        self.session.update_id(self.selected, track_id)
        self._render()

    def _delete_selected(self):
        if self.session is None or self.selected is None:
            return
        self.session.delete_box(self.selected)
        self.selected = None
        self._render()

    def _delete_id(self):
        from tkinter import messagebox
        if self.session is None:
            return
        try:
            track_id = int(self.id_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter a numeric ID")
            return
        count = self.session.delete_id(track_id)
        self.selected = None
        self._render()
        self.status.config(text=f"Deleted {count} boxes with ID {track_id}")

    def _rename_all(self):
        from tkinter import messagebox
        if self.session is None:
            return
        try:
            source_id = int(self.src_entry.get())
            target_id = int(self.dst_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Enter numeric source and target IDs")
            return
        count = self.session.rename_id(source_id, target_id)
        self.selected = None
        self._render()
        self.status.config(text=f"Renamed {count} boxes {source_id} -> {target_id}")

    def _propagate(self):
        from tkinter import messagebox
        if self.session is None:
            return
        try:
            track_id = int(self.bulk_id_entry.get())
            final_frame = int(self.bulk_final_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Enter numeric ID and final frame")
            return
        count = self.session.propagate(track_id, final_frame)
        self._render()
        self.status.config(text=f"Propagated {count} boxes for ID {track_id}")

    def _undo(self):
        if self.session and self.session.undo():
            self.selected = None
            self._render()
            self.status.config(text="Undo")

    def _save(self):
        if self.session:
            self.session.save()
            self.status.config(text="Saved")

    def _reset(self):
        from tkinter import messagebox
        if self.session and messagebox.askyesno("Reset", "Reload original annotations?"):
            self.session.reload()
            self.selected = None
            self._render()

    def run(self):
        self.window.mainloop()


def launch(directory: Optional[str] = None, auto_save: bool = True) -> None:
    ReviewApp(directory=directory, auto_save=auto_save).run()
