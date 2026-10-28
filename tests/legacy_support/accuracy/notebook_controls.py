"""Responsive notebook controls. Training runs in one worker so Pause can be clicked."""
from pathlib import Path
import threading


class TrainingSession:
    def __init__(self, data, run, device=0, runner=None, changed=None):
        self.data, self.run, self.device = Path(data), Path(run), device
        self.runner = runner
        self.changed = changed or (lambda: None)
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self.thread = None
        self.status = 'ready'
        self.error = None
        self.completed_epochs = None
        self.archive = self.run.parent / 'six_class_recovery_checkpoint.zip'

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def pause_requested(self):
        return self._pause.is_set()

    def pause(self):
        if self.running:
            self._pause.set()
            self.status = 'Pausing after the current epoch finishes and its checkpoint is saved…'
            self.changed()

    def checkpoint_saved(self, epoch, archive):
        self.completed_epochs, self.archive = epoch, Path(archive)
        self.changed()

    def start(self, resume=False):
        with self._lock:
            if self.running:
                raise RuntimeError('Training is already running. Do not launch a second copy.')
            if resume and not (self.archive.exists() or (self.run/'training/weights/last.pt').exists()):
                raise RuntimeError('Resume refused: no checkpoint. Attach the downloaded ZIP and rerun setup.')
            if not resume and ((self.run/'training/weights/last.pt').exists() or self.archive.exists()):
                raise RuntimeError('A checkpoint already exists. Click Resume instead of starting over.')
            self._pause.clear()
            self.error = None
            self.status = 'resuming' if resume else 'running'
            self.thread = threading.Thread(target=self._work, args=(resume,), name='pcb-training', daemon=True)
            self.thread.start()
        self.changed()

    def _work(self, resume):
        try:
            if self.runner is None:
                from train_recovery import main
                runner = main
            else:
                runner = self.runner
            result = runner(self.data, self.run, device=self.device, control=self, resume_only=resume)
            self.status = result['status']
        except BaseException as error:
            self.error = str(error)
            self.status = 'error'
        finally:
            self.changed()


def show_controls(data, run, device=0):
    import html
    import ipywidgets as widgets
    from IPython.display import display, FileLink

    start = widgets.Button(description='Start training', button_style='success')
    pause = widgets.Button(description='Pause', button_style='warning', disabled=True)
    resume = widgets.Button(description='Resume', button_style='info')
    status = widgets.HTML()
    download = widgets.HTML()
    messages = widgets.Output()
    loop = getattr(getattr(__import__('IPython').get_ipython(), 'kernel', None), 'io_loop', None)

    def render():
        # The worker's finally callback runs before thread.is_alive() flips false.
        active = session.status in {'running', 'resuming'} or session.status.startswith('Pausing')
        start.disabled = active or session.archive.exists() or (session.run/'training/weights/last.pt').exists()
        pause.disabled = not active or session.pause_requested()
        resume.disabled = active or not (session.archive.exists() or (session.run/'training/weights/last.pt').exists())
        text = html.escape(session.status)
        if session.completed_epochs is not None:
            text += f' · checkpoint: epoch {session.completed_epochs}'
        if session.error:
            text += '<br>' + html.escape(session.error)
        status.value = '<b>' + text + '</b>'
        if session.archive.exists():
            download.value = FileLink(str(session.archive))._repr_html_() + '<br>Download this ZIP before ending/resetting the session.'
        else:
            download.value = 'A checkpoint download appears after the first completed epoch.'
        results = session.run.parent/'six_class_recovery_results.zip'
        if session.status == 'complete' and results.exists():
            download.value += '<br>Final results: ' + FileLink(str(results))._repr_html_()

    def changed():
        if loop is not None:
            loop.add_callback(render)
        else:
            render()

    session = TrainingSession(data, run, device, changed=changed)

    def launch(resuming):
        try:
            session.start(resume=resuming)
        except RuntimeError as error:
            with messages:
                print(error)

    start.on_click(lambda _: launch(False))
    pause.on_click(lambda _: session.pause())
    resume.on_click(lambda _: launch(True))
    render()
    display(widgets.VBox([widgets.HBox([start, pause, resume]), status, download, messages]))
    return session
