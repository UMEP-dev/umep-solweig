# umep-solweig

A python package to run the SOLWEIG algorithm.

## How to install

To install the package, run pip install umep-solweig

If you want some execution examples, look at the files inside the tests folder of the github.

## how to build the package from the source code :

### Setup

To build you package you need to install the build pip package inside you venv.

```bash
pip install build
```

### Building

The to build the plugin you can run :

```bash
python -m build
```

### How to install the version you built

Run inside you terminal from the project's root :

```bash
pip install ./dist/umep_solweig-{version_year}.{version}-py{python_major_version}-none-any.whl
```

If you already have the package installed, do not forger to uninstall the current version before installing the new one with :

```bash
pip uninstall umep-solweig -y
```

## Development

If you're working on the code, please do not forger to run the tests after you've build and installed your local version of the package. **This is a realy important step, do not avoid it**. From the root run :

```bash
python -m pytest -q ./tests/
```

Ps: a new algorithm added or a new feature added means a new test added. If you've change the way an algorithm is working, please update the tests.
