#! /usr/bin/env python

# Used these links for some inspiration:
# 
# * get_build_dir():
#   https://github.com/symengine/symengine.py/blob/master/setup.py
#
# * custom build process:
#   https://stackoverflow.com/questions/3517301/python-distutils-writing-c-extentions-with-generated-source-code
#
# * setup syntax:
#   https://github.com/dsuch/globre-lgpl/blob/master/setup.py
# 
# * git tag versioning:
#   http://blogs.nopcode.org/brainstorm/2013/05/20/pragmatic-python-versioning-via-setuptools-and-git-tags/
#   numpy setup.py

from setuptools import setup, find_packages,Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy, os, sys, subprocess

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--tags', '--dirty'])
        version = out.strip().decode('ascii')

        if "-" in version:
            version,sig = version.split("-",1)
            version+=".dev"
            if "-" in sig:
                sig,_ = sig.split("-",1)
                version+=sig

        #some kind of mechanism to tag a dev version...
        #if "dirty" in version:
        #    version = version.replace("dirty","dev")
        if version == '':
            version = "Unknown"
    except OSError:
        version = "Unknown"

    return version

def read(*parts, **kw):
  heredir = os.path.abspath(os.path.dirname(__file__))
  try:    return open(os.path.join(heredir, *parts)).read()
  except: return kw.get('default', '')

def get_build_dir(dist):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    build = dist.get_command_obj('build')
    build_ext = dist.get_command_obj('build_ext')
    return source_dir if build_ext.inplace else build.build_platlib

def change_swig(fn):
    with open(fn) as f:
        lines = f.readlines()

    new_lines = [
        '        pkg_parts = __name__.rpartition(\'.\')\n',
        '        pkg = pkg_parts[0]\n',
        '        if pkg_parts[1] != \'.\':\n',
        '            pkg = pkg_parts[2]\n'
    ]
    substr = 'pkg = __name__.rpartition(\'.\')[0]'

    indices = [i for i, s in enumerate(lines) if substr in s]
    if len(indices) > 0:
        lines = lines[:indices[0]]+new_lines+lines[indices[0]+1:]

    return lines

class build_ext(_build_ext):
    description = 'Custom Build Process'

    def initialize_options(self):
        _build_ext.initialize_options(self)
    def finalize_options(self):
        _build_ext.finalize_options(self)

    def run(self):
        # Start classic Extension build
        _build_ext.run(self)
        print "==="
        print self.distribution.metadata.version
        print self.distribution.metadata.release
        print "==="
        # 1. Rename the swig generated .py file to __init__.py
        #
        # 2. if not inplace, additionally move all the files to
        #    a subdirectory of ext.name in build_dir
        #
        # 3. add a version.py file

        source_dir = os.path.dirname(os.path.realpath(__file__))
        build_dir = get_build_dir(self.distribution)

        for ext in self.extensions:
            name = ext.name
            if name.startswith('_'): name = name[1:]

            fn = os.path.join(source_dir,name+'.py')
            if os.path.exists(fn):
                lines = change_swig(fn)
                if self.inplace:
                    os.remove(fn)
                    build_dir = source_dir
                else:
                    build_dir = os.path.join(self.build_lib,name)
                    if not os.path.exists(build_dir):
                        os.mkdir(build_dir)
                    #We need to move the file...
                    fn_old = self.get_ext_fullpath(ext.name)
                    fn_new = os.path.join(build_dir,
                        os.path.split(fn_old)[1])
                    os.rename(fn_old,fn_new)

                fn = os.path.join(build_dir,'__init__.py')
                with open(fn,'w') as f:
                    f.writelines(lines)

                fn = os.path.join(build_dir,'version.py')
                version_msg = '# Do not edit this file, pipeline versioning is governed by git tags'
                version = self.distribution.metadata.version
                with open(fn, 'w') as fh:
                    fh.write(version_msg + os.linesep + '__version__ = \'' + version +'\'')

# atrous extension module
_extension = Extension('_libatrous',
                   ['libatrous.i','matalloc.c','libatrous.c'],
                   include_dirs = [numpy_include],

                   extra_compile_args = ['--verbose','-march=native','-O3','-ftree-vectorizer-verbose=2','-ffast-math'],
                   swig_opts=['-builtin'],
                   extra_link_args=[],
                   )

_extension.extra_compile_args.append('-fopenmp')
_extension.extra_link_args.append('-lgomp')


test_dependencies = [
]

dependencies = [
]

entrypoints = {
}

classifiers = [
  'Development Status :: 4 - Beta',
  # 'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Developers',
  'Intended Audience :: Science/Research',
  'Programming Language :: C',
  'Programming Language :: Python',
  'Programming Language :: Python :: 2',
  'Programming Language :: Python :: 2.7',
  'Topic :: Software Development',
  'Topic :: Scientific/Engineering',
  'Operating System :: Microsoft :: Windows',
  'Operating System :: POSIX',
  'Operating System :: Unix',
  'Natural Language :: English',
  'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
]

# NumyTypemapTests setup
setup(
    name                  = 'libatrous',
    #version               = read('../VERSION.txt', default='0.0.1').strip(),
    version               = git_version(),
    classifiers           = classifiers,
    description           = 'Atrous wavelet library (1d, 2d, 3d)',
    long_description      = read('../README.md'),
    url                   = 'https://github.com/zindy/libatrous',
    keywords              = 'python imaging atrous wavelet microscopy',
    author                = 'Egor Zindy',
    author_email          = 'ezindy@gmail.com',
    packages              = find_packages(),
    platforms             = ['any'],
    include_package_data  = True,
    zip_safe              = True,
    release               = False,
    #install_requires      = dependencies,
    #tests_require         = test_dependencies,
    #test_suite            = 'libatrous',
    entry_points          = entrypoints,
    license               = 'LGPLv3+',

    ext_modules  = [_extension],
    cmdclass = { 'build_ext': build_ext},
)

