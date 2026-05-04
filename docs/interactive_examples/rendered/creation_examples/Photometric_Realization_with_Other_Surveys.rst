Photometric error stage demo
============================

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel.

If you’re interested in running this in pipeline mode, see
`02_Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/creation_examples/02_Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``pipeline_examples/core_examples/`` folder.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from rail.core.data import PqHandle
    from rail.interactive.creation.degraders import photometric_errors


.. parsed-literal::

    Install FSPS with the following commands:
    pip uninstall fsps
    git clone --recursive https://github.com/dfm/python-fsps.git
    cd python-fsps
    python -m pip install .
    export SPS_HOME=$(pwd)/src/fsps/libfsps
    
    LEPHAREDIR is being set to the default cache directory:
    /home/runner/.cache/lephare/data
    More than 1Gb may be written there.
    LEPHAREWORK is being set to the default cache directory:
    /home/runner/.cache/lephare/work
    Default work cache is already linked. 
    This is linked to the run directory:
    /home/runner/.cache/lephare/runs/20260504T122301


.. parsed-literal::

    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/runpy.py", line 196, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/runpy.py", line 86, in _run_code
        exec(code, run_globals)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel_launcher.py", line 18, in <module>
        app.launch_new_instance()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/traitlets/config/application.py", line 1075, in launch_instance
        app.start()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelapp.py", line 758, in start
        self.io_loop.start()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/tornado/platform/asyncio.py", line 211, in start
        self.asyncio_loop.run_forever()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/base_events.py", line 603, in run_forever
        self._run_once()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/base_events.py", line 1909, in _run_once
        handle._run()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/events.py", line 80, in _run
        self._context.run(self._callback, *self._args)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/utils.py", line 71, in preserve_context
        return await f(*args, **kwargs)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 621, in shell_main
        await self.dispatch_shell(msg, subshell_id=subshell_id)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 478, in dispatch_shell
        await result
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 372, in execute_request
        await super().execute_request(stream, ident, parent)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 834, in execute_request
        reply_content = await reply_content
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 464, in do_execute
        res = shell.run_cell(
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/zmqshell.py", line 663, in run_cell
        return super().run_cell(*args, **kwargs)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3077, in run_cell
        result = self._run_cell(
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3132, in _run_cell
        result = runner(coro)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/async_helpers.py", line 128, in _pseudo_sync_runner
        coro.send(None)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3336, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3519, in run_ast_nodes
        if await self.run_code(code, result, async_=asy):
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3579, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "/tmp/ipykernel_5884/2313627096.py", line 5, in <module>
        from rail.interactive.creation.degraders import photometric_errors
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/interactive/__init__.py", line 3, in <module>
        from . import calib, creation, estimation, evaluation, tools
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/interactive/calib/__init__.py", line 3, in <module>
        from rail.utils.interactive.initialize_utils import _initialize_interactive_module
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/utils/interactive/initialize_utils.py", line 17, in <module>
        from rail.utils.interactive.base_utils import (
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/utils/interactive/base_utils.py", line 10, in <module>
        rail.stages.import_and_attach_all(silent=True)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/stages/__init__.py", line 74, in import_and_attach_all
        RailEnv.import_all_packages(silent=silent)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/core/introspection.py", line 541, in import_all_packages
        _imported_module = importlib.import_module(pkg)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/som/__init__.py", line 1, in <module>
        from rail.creation.degraders.specz_som import *
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/creation/degraders/specz_som.py", line 15, in <module>
        from somoclu import Somoclu
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/__init__.py", line 11, in <module>
        from .train import Somoclu
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/train.py", line 25, in <module>
        from .somoclu_wrap import train as wrap_train
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/somoclu_wrap.py", line 11, in <module>
        import _somoclu_wrap


::


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    File /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/numpy/core/_multiarray_umath.py:44, in __getattr__(attr_name)
         39     # Also print the message (with traceback).  This is because old versions
         40     # of NumPy unfortunately set up the import to replace (and hide) the
         41     # error.  The traceback shouldn't be needed, but e.g. pytest plugins
         42     # seem to swallow it and we should be failing anyway...
         43     sys.stderr.write(msg + tb_msg)
    ---> 44     raise ImportError(msg)
         46 ret = getattr(_multiarray_umath, attr_name, None)
         47 if ret is None:


    ImportError: 
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    



.. parsed-literal::

    Warning: the binary library cannot be imported. You cannot train maps, but you can load and analyze ones that you have already saved.
    The problem occurs because either compilation failed when you installed Somoclu or a path is missing from the dependencies when you are trying to import it. Please refer to the documentation to see your options.


Create a random catalog with ugrizy+YJHF bands as the the true input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    data = np.random.normal(23, 3, size=(1000, 9))
    
    data_df = pd.DataFrame(
        data=data, columns=["u", "g", "r", "i", "z", "y", "Y", "J", "H"]  # values
    )
    data_truth = PqHandle("input")
    data_truth.set_data(data_df)

.. code:: ipython3

    data_df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>Y</th>
          <th>J</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.708673</td>
          <td>23.946529</td>
          <td>21.793153</td>
          <td>22.897256</td>
          <td>27.120461</td>
          <td>22.105663</td>
          <td>27.084293</td>
          <td>20.308320</td>
          <td>25.756885</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.282685</td>
          <td>26.173883</td>
          <td>21.843843</td>
          <td>23.229807</td>
          <td>27.365256</td>
          <td>22.150652</td>
          <td>18.333430</td>
          <td>24.662477</td>
          <td>21.588986</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.068729</td>
          <td>24.991553</td>
          <td>17.721741</td>
          <td>21.665644</td>
          <td>19.694103</td>
          <td>26.530101</td>
          <td>22.851100</td>
          <td>18.312128</td>
          <td>22.045591</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.263871</td>
          <td>19.091246</td>
          <td>23.058038</td>
          <td>22.922223</td>
          <td>28.248949</td>
          <td>24.605646</td>
          <td>27.281578</td>
          <td>21.562341</td>
          <td>22.428945</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.436487</td>
          <td>22.040293</td>
          <td>20.639063</td>
          <td>23.339375</td>
          <td>26.307421</td>
          <td>13.015766</td>
          <td>16.941755</td>
          <td>22.432366</td>
          <td>21.820664</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>27.501988</td>
          <td>27.510125</td>
          <td>20.514106</td>
          <td>23.338441</td>
          <td>24.727934</td>
          <td>21.107117</td>
          <td>23.330389</td>
          <td>24.265659</td>
          <td>20.295622</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.373435</td>
          <td>27.492324</td>
          <td>23.963002</td>
          <td>19.532881</td>
          <td>22.863382</td>
          <td>22.223532</td>
          <td>21.672254</td>
          <td>23.335300</td>
          <td>23.306935</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.570164</td>
          <td>20.383393</td>
          <td>22.535414</td>
          <td>23.988333</td>
          <td>25.713129</td>
          <td>24.180628</td>
          <td>21.961490</td>
          <td>24.973341</td>
          <td>21.687583</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.060170</td>
          <td>22.091996</td>
          <td>22.182697</td>
          <td>27.933735</td>
          <td>24.864371</td>
          <td>22.403997</td>
          <td>19.816141</td>
          <td>20.898465</td>
          <td>29.882491</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.301085</td>
          <td>23.441800</td>
          <td>24.427021</td>
          <td>22.399240</td>
          <td>18.251258</td>
          <td>24.263637</td>
          <td>17.975266</td>
          <td>21.319901</td>
          <td>20.215627</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 9 columns</p>
    </div>



The LSST error model adds noise to the optical bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    samples_w_errs = photometric_errors.lsst_error_model(sample=data_truth)
    
    samples_w_errs["output"]


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>Y</th>
          <th>J</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.630824</td>
          <td>0.185991</td>
          <td>23.933616</td>
          <td>0.014941</td>
          <td>21.793131</td>
          <td>0.005341</td>
          <td>22.893709</td>
          <td>0.009361</td>
          <td>27.056231</td>
          <td>0.543040</td>
          <td>22.087938</td>
          <td>0.017699</td>
          <td>27.084293</td>
          <td>20.308320</td>
          <td>25.756885</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.277026</td>
          <td>0.006590</td>
          <td>26.292556</td>
          <td>0.115721</td>
          <td>21.840965</td>
          <td>0.005368</td>
          <td>23.241852</td>
          <td>0.011920</td>
          <td>26.938816</td>
          <td>0.498349</td>
          <td>22.165223</td>
          <td>0.018880</td>
          <td>18.333430</td>
          <td>24.662477</td>
          <td>21.588986</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.171228</td>
          <td>0.125598</td>
          <td>25.007299</td>
          <td>0.037266</td>
          <td>17.722837</td>
          <td>0.005002</td>
          <td>21.670283</td>
          <td>0.005681</td>
          <td>19.693391</td>
          <td>0.005096</td>
          <td>26.516532</td>
          <td>0.708515</td>
          <td>22.851100</td>
          <td>18.312128</td>
          <td>22.045591</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.181111</td>
          <td>0.052799</td>
          <td>19.094522</td>
          <td>0.005013</td>
          <td>23.049139</td>
          <td>0.007471</td>
          <td>22.928482</td>
          <td>0.009574</td>
          <td>28.274005</td>
          <td>1.189828</td>
          <td>24.402719</td>
          <td>0.134246</td>
          <td>27.281578</td>
          <td>21.562341</td>
          <td>22.428945</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.437764</td>
          <td>0.027606</td>
          <td>22.047024</td>
          <td>0.005702</td>
          <td>20.636580</td>
          <td>0.005057</td>
          <td>23.304321</td>
          <td>0.012491</td>
          <td>26.045647</td>
          <td>0.247485</td>
          <td>13.017342</td>
          <td>0.005000</td>
          <td>16.941755</td>
          <td>22.432366</td>
          <td>21.820664</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>27.705515</td>
          <td>0.884704</td>
          <td>27.504642</td>
          <td>0.319840</td>
          <td>20.505955</td>
          <td>0.005047</td>
          <td>23.329722</td>
          <td>0.012735</td>
          <td>24.727396</td>
          <td>0.079824</td>
          <td>21.108153</td>
          <td>0.008633</td>
          <td>23.330389</td>
          <td>24.265659</td>
          <td>20.295622</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.392198</td>
          <td>0.026548</td>
          <td>27.346100</td>
          <td>0.281577</td>
          <td>23.985048</td>
          <td>0.013780</td>
          <td>19.529388</td>
          <td>0.005026</td>
          <td>22.844730</td>
          <td>0.015489</td>
          <td>22.247641</td>
          <td>0.020240</td>
          <td>21.672254</td>
          <td>23.335300</td>
          <td>23.306935</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.568733</td>
          <td>0.007380</td>
          <td>20.385149</td>
          <td>0.005062</td>
          <td>22.542435</td>
          <td>0.006135</td>
          <td>23.996747</td>
          <td>0.021972</td>
          <td>25.413234</td>
          <td>0.145234</td>
          <td>24.265377</td>
          <td>0.119180</td>
          <td>21.961490</td>
          <td>24.973341</td>
          <td>21.687583</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.907818</td>
          <td>0.234374</td>
          <td>22.090974</td>
          <td>0.005751</td>
          <td>22.182216</td>
          <td>0.005638</td>
          <td>27.906721</td>
          <td>0.590850</td>
          <td>24.838217</td>
          <td>0.088014</td>
          <td>22.395117</td>
          <td>0.022960</td>
          <td>19.816141</td>
          <td>20.898465</td>
          <td>29.882491</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.295356</td>
          <td>1.251526</td>
          <td>23.441307</td>
          <td>0.010367</td>
          <td>24.420333</td>
          <td>0.019672</td>
          <td>22.386329</td>
          <td>0.007097</td>
          <td>18.252852</td>
          <td>0.005013</td>
          <td>24.272939</td>
          <td>0.119966</td>
          <td>17.975266</td>
          <td>21.319901</td>
          <td>20.215627</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "ugrizy":
        # pull out the magnitudes and errors
        mags = samples_w_errs["output"][band].to_numpy()
        errs = samples_w_errs["output"][band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()



.. image:: Photometric_Realization_with_Other_Surveys_files/Photometric_Realization_with_Other_Surveys_7_0.png


The Roman error model adds noise to the infrared bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    samples_w_errs_roman = photometric_errors.roman_error_model(
        sample=data_truth,
        m5={"Y": 27.0, "J": 26.4, "H": 26.4},
        theta={"Y": 27.0, "J": 0.106, "H": 0.128},
    )
    
    samples_w_errs_roman["output"]


.. parsed-literal::

    Inserting handle into data store.  input: None, RomanErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, RomanErrorModel




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>Y</th>
          <th>Y_err</th>
          <th>J</th>
          <th>J_err</th>
          <th>H</th>
          <th>H_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.708673</td>
          <td>23.946529</td>
          <td>21.793153</td>
          <td>22.897256</td>
          <td>27.120461</td>
          <td>22.105663</td>
          <td>27.061618</td>
          <td>0.208514</td>
          <td>20.311058</td>
          <td>0.005063</td>
          <td>25.727127</td>
          <td>0.111071</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.282685</td>
          <td>26.173883</td>
          <td>21.843843</td>
          <td>23.229807</td>
          <td>27.365256</td>
          <td>22.150652</td>
          <td>18.335422</td>
          <td>0.005001</td>
          <td>24.655146</td>
          <td>0.042959</td>
          <td>21.588569</td>
          <td>0.005624</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.068729</td>
          <td>24.991553</td>
          <td>17.721741</td>
          <td>21.665644</td>
          <td>19.694103</td>
          <td>26.530101</td>
          <td>22.858246</td>
          <td>0.006908</td>
          <td>18.300880</td>
          <td>0.005002</td>
          <td>22.044315</td>
          <td>0.006351</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.263871</td>
          <td>19.091246</td>
          <td>23.058038</td>
          <td>22.922223</td>
          <td>28.248949</td>
          <td>24.605646</td>
          <td>27.133414</td>
          <td>0.221400</td>
          <td>21.562315</td>
          <td>0.005596</td>
          <td>22.426819</td>
          <td>0.007482</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.436487</td>
          <td>22.040293</td>
          <td>20.639063</td>
          <td>23.339375</td>
          <td>26.307421</td>
          <td>13.015766</td>
          <td>16.945354</td>
          <td>0.005000</td>
          <td>22.428970</td>
          <td>0.007491</td>
          <td>21.829336</td>
          <td>0.005943</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>27.501988</td>
          <td>27.510125</td>
          <td>20.514106</td>
          <td>23.338441</td>
          <td>24.727934</td>
          <td>21.107117</td>
          <td>23.351937</td>
          <td>0.009018</td>
          <td>24.296531</td>
          <td>0.031232</td>
          <td>20.290715</td>
          <td>0.005060</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.373435</td>
          <td>27.492324</td>
          <td>23.963002</td>
          <td>19.532881</td>
          <td>22.863382</td>
          <td>22.223532</td>
          <td>21.677569</td>
          <td>0.005252</td>
          <td>23.336396</td>
          <td>0.013772</td>
          <td>23.291633</td>
          <td>0.013292</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.570164</td>
          <td>20.383393</td>
          <td>22.535414</td>
          <td>23.988333</td>
          <td>25.713129</td>
          <td>24.180628</td>
          <td>21.958200</td>
          <td>0.005416</td>
          <td>24.945464</td>
          <td>0.055647</td>
          <td>21.684830</td>
          <td>0.005737</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.060170</td>
          <td>22.091996</td>
          <td>22.182697</td>
          <td>27.933735</td>
          <td>24.864371</td>
          <td>22.403997</td>
          <td>19.812257</td>
          <td>0.005008</td>
          <td>20.911882</td>
          <td>0.005187</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.301085</td>
          <td>23.441800</td>
          <td>24.427021</td>
          <td>22.399240</td>
          <td>18.251258</td>
          <td>24.263637</td>
          <td>17.969783</td>
          <td>0.005000</td>
          <td>21.308598</td>
          <td>0.005381</td>
          <td>20.216587</td>
          <td>0.005053</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 12 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "YJH":
        # pull out the magnitudes and errors
        mags = samples_w_errs_roman["output"][band].to_numpy()
        errs = samples_w_errs_roman["output"][band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()



.. image:: Photometric_Realization_with_Other_Surveys_files/Photometric_Realization_with_Other_Surveys_10_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    samples_w_errs_Euclid = photometric_errors.euclid_error_model(sample=data_truth)
    
    samples_w_errs_Euclid["output"]


.. parsed-literal::

    Inserting handle into data store.  input: None, EuclidErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, EuclidErrorModel




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>Y</th>
          <th>Y_err</th>
          <th>J</th>
          <th>J_err</th>
          <th>H</th>
          <th>H_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.708673</td>
          <td>23.946529</td>
          <td>21.793153</td>
          <td>22.897256</td>
          <td>27.120461</td>
          <td>22.105663</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.326094</td>
          <td>0.006813</td>
          <td>25.779062</td>
          <td>0.583951</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.282685</td>
          <td>26.173883</td>
          <td>21.843843</td>
          <td>23.229807</td>
          <td>27.365256</td>
          <td>22.150652</td>
          <td>18.325968</td>
          <td>0.005077</td>
          <td>24.105335</td>
          <td>0.141428</td>
          <td>21.591531</td>
          <td>0.016962</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.068729</td>
          <td>24.991553</td>
          <td>17.721741</td>
          <td>21.665644</td>
          <td>19.694103</td>
          <td>26.530101</td>
          <td>22.927314</td>
          <td>0.059855</td>
          <td>18.320692</td>
          <td>0.005053</td>
          <td>22.048303</td>
          <td>0.025100</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.263871</td>
          <td>19.091246</td>
          <td>23.058038</td>
          <td>22.922223</td>
          <td>28.248949</td>
          <td>24.605646</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.574051</td>
          <td>0.015391</td>
          <td>22.429592</td>
          <td>0.035145</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.436487</td>
          <td>22.040293</td>
          <td>20.639063</td>
          <td>23.339375</td>
          <td>26.307421</td>
          <td>13.015766</td>
          <td>16.949229</td>
          <td>0.005006</td>
          <td>22.428838</td>
          <td>0.032139</td>
          <td>21.830092</td>
          <td>0.020768</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>27.501988</td>
          <td>27.510125</td>
          <td>20.514106</td>
          <td>23.338441</td>
          <td>24.727934</td>
          <td>21.107117</td>
          <td>23.276041</td>
          <td>0.081567</td>
          <td>24.357426</td>
          <td>0.175509</td>
          <td>20.286723</td>
          <td>0.006996</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.373435</td>
          <td>27.492324</td>
          <td>23.963002</td>
          <td>19.532881</td>
          <td>22.863382</td>
          <td>22.223532</td>
          <td>21.648882</td>
          <td>0.019373</td>
          <td>23.474569</td>
          <td>0.081461</td>
          <td>23.310696</td>
          <td>0.076985</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.570164</td>
          <td>20.383393</td>
          <td>22.535414</td>
          <td>23.988333</td>
          <td>25.713129</td>
          <td>24.180628</td>
          <td>21.966516</td>
          <td>0.025503</td>
          <td>25.183845</td>
          <td>0.346145</td>
          <td>21.674827</td>
          <td>0.018192</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.060170</td>
          <td>22.091996</td>
          <td>22.182697</td>
          <td>27.933735</td>
          <td>24.864371</td>
          <td>22.403997</td>
          <td>19.812834</td>
          <td>0.006086</td>
          <td>20.897341</td>
          <td>0.009286</td>
          <td>27.763292</td>
          <td>1.836884</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.301085</td>
          <td>23.441800</td>
          <td>24.427021</td>
          <td>22.399240</td>
          <td>18.251258</td>
          <td>24.263637</td>
          <td>17.971843</td>
          <td>0.005040</td>
          <td>21.299707</td>
          <td>0.012376</td>
          <td>20.214073</td>
          <td>0.006779</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 12 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "YJH":
        # pull out the magnitudes and errors
        mags = samples_w_errs_Euclid["output"][band].to_numpy()
        errs = samples_w_errs_Euclid["output"][band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()



.. image:: Photometric_Realization_with_Other_Surveys_files/Photometric_Realization_with_Other_Surveys_13_0.png

