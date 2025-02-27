
import os
import glob

from absl import app
from tqdm import tqdm
from absl import flags
from gfootball.env import script_helpers

FLAGS = flags.FLAGS

flags.DEFINE_string('trace_file', None, 'Trace file to convert')
flags.DEFINE_string('output', None, 'Output txt file')
flags.DEFINE_bool('include_debug', True,
                  'Include debug information for each step')

def main(_):
  dump_files = glob.glob('./dump/*.dump')
  for dump_file_dir in tqdm(dump_files, desc="Processing Dump Files", unit="file"):
    file_dir = dump_file_dir.replace('/dump/', '/txt/').replace(".dump", ".txt")
    script_helpers.ScriptHelpers().dump_to_txt(dump_file_dir, file_dir,FLAGS.include_debug)
    
if __name__ == '__main__':
  app.run(main)