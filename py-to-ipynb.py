#Source: https://www.webucator.com/blog/2015/07/bulk-convert-python-files-to-ipython-notebook-files-py-to-ipynb-conversion/s

import os
from json.encoder import JSONEncoder

nb_start = '''{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": ['''

nb_end =''']
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.4.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}'''

def main():
    for dirpath, dirnames, filenames in os.walk('.'):
        for fname in filenames:
            path = dirpath+'/'+fname
            if fname[-3:] == '.py' and not os.path.samefile(path,'.\\py-to-ipynb.py'):
                path = dirpath+'/'+fname
                nb_path = path[:-3] + '.ipynb'
                with open(path,'r') as f_in:
                    f_in_content = JSONEncoder().encode(f_in.read())
                    nb_content = nb_start + f_in_content + nb_end

                    with open(nb_path,'w') as f_out:
                        f_out.write(nb_content)

                print("Created",nb_path)

main()
