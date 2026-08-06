"""Deprecated LAPO entry point forwarding to the unified pixel runner."""

from absl import flags

from train_pixel import run


flags.FLAGS.set_default('algorithm', 'gc_pixel_lapo_decoder')


if __name__ == '__main__':
    run()
