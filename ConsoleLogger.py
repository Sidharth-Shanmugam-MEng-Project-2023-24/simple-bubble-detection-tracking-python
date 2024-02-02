import consoledraw as CD

class CDLogger:
    
    def __init__(self):
        self.console = CD.Console()
        self.format = """
        Tot. frames processed: {}
        Avg. FPT: {} ms
            Tgt. FPT: {} ms
        Avg. FPS: {} fps
            Tgt. FPS: {} fps
        """

    def display(self, nframes, afpt, tfpt, afps, tfps):
        with self.console:
            self.console.print(
                self.format.format(
                    nframes,
                    afpt,
                    tfpt,
                    afps,
                    tfps
                )
            )
