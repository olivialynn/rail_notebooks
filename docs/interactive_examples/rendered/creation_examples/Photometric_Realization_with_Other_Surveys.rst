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
    /home/runner/.cache/lephare/runs/20260601T134643


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
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/traitlets/config/application.py", line 1082, in launch_instance
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
      File "/tmp/ipykernel_4040/2313627096.py", line 5, in <module>
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
          <td>27.597138</td>
          <td>23.410746</td>
          <td>24.332932</td>
          <td>21.738095</td>
          <td>25.762221</td>
          <td>22.901580</td>
          <td>22.856230</td>
          <td>24.178991</td>
          <td>21.208914</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.492572</td>
          <td>24.849699</td>
          <td>21.856477</td>
          <td>28.348929</td>
          <td>24.810721</td>
          <td>24.516700</td>
          <td>25.210521</td>
          <td>25.491282</td>
          <td>26.925990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.528775</td>
          <td>24.695894</td>
          <td>19.489144</td>
          <td>22.534017</td>
          <td>22.105033</td>
          <td>28.391906</td>
          <td>27.913236</td>
          <td>23.519688</td>
          <td>23.033575</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.001564</td>
          <td>25.114046</td>
          <td>25.460844</td>
          <td>13.027881</td>
          <td>29.276571</td>
          <td>16.347359</td>
          <td>18.821405</td>
          <td>25.464906</td>
          <td>28.687989</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.741741</td>
          <td>19.055497</td>
          <td>20.419327</td>
          <td>21.201373</td>
          <td>19.648462</td>
          <td>22.229420</td>
          <td>20.813919</td>
          <td>24.404762</td>
          <td>23.080636</td>
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
          <td>20.784500</td>
          <td>22.244543</td>
          <td>22.330757</td>
          <td>23.481757</td>
          <td>25.612422</td>
          <td>25.472825</td>
          <td>23.737939</td>
          <td>20.384226</td>
          <td>22.815083</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.693647</td>
          <td>27.168555</td>
          <td>26.823348</td>
          <td>23.964394</td>
          <td>19.816125</td>
          <td>22.723520</td>
          <td>25.730922</td>
          <td>23.142357</td>
          <td>26.550749</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.858594</td>
          <td>22.220576</td>
          <td>20.974283</td>
          <td>20.229863</td>
          <td>21.850657</td>
          <td>26.088833</td>
          <td>11.442345</td>
          <td>20.401696</td>
          <td>28.636086</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.300828</td>
          <td>26.489568</td>
          <td>23.922031</td>
          <td>26.914451</td>
          <td>21.419059</td>
          <td>22.900724</td>
          <td>20.922308</td>
          <td>25.153080</td>
          <td>23.993964</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.449172</td>
          <td>24.369495</td>
          <td>18.990733</td>
          <td>24.942895</td>
          <td>22.184028</td>
          <td>22.599188</td>
          <td>22.904663</td>
          <td>23.185487</td>
          <td>21.214893</td>
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
          <td>30.102427</td>
          <td>2.737282</td>
          <td>23.407993</td>
          <td>0.010137</td>
          <td>24.338807</td>
          <td>0.018368</td>
          <td>21.726908</td>
          <td>0.005746</td>
          <td>25.986121</td>
          <td>0.235627</td>
          <td>22.916062</td>
          <td>0.036208</td>
          <td>22.856230</td>
          <td>24.178991</td>
          <td>21.208914</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.503096</td>
          <td>0.005148</td>
          <td>24.840171</td>
          <td>0.032171</td>
          <td>21.854077</td>
          <td>0.005376</td>
          <td>27.906369</td>
          <td>0.590702</td>
          <td>24.872526</td>
          <td>0.090710</td>
          <td>24.621745</td>
          <td>0.162039</td>
          <td>25.210521</td>
          <td>25.491282</td>
          <td>26.925990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.652200</td>
          <td>0.855331</td>
          <td>24.713740</td>
          <td>0.028803</td>
          <td>19.490800</td>
          <td>0.005012</td>
          <td>22.535723</td>
          <td>0.007619</td>
          <td>22.094043</td>
          <td>0.008986</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.913236</td>
          <td>23.519688</td>
          <td>23.033575</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.004021</td>
          <td>0.019162</td>
          <td>25.043191</td>
          <td>0.038465</td>
          <td>25.430378</td>
          <td>0.047588</td>
          <td>13.028362</td>
          <td>0.005000</td>
          <td>27.492445</td>
          <td>0.736039</td>
          <td>16.353453</td>
          <td>0.005004</td>
          <td>18.821405</td>
          <td>25.464906</td>
          <td>28.687989</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.743195</td>
          <td>0.005062</td>
          <td>19.063674</td>
          <td>0.005013</td>
          <td>20.416982</td>
          <td>0.005042</td>
          <td>21.203254</td>
          <td>0.005320</td>
          <td>19.656611</td>
          <td>0.005091</td>
          <td>22.246399</td>
          <td>0.020219</td>
          <td>20.813919</td>
          <td>24.404762</td>
          <td>23.080636</td>
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
          <td>20.793837</td>
          <td>0.005809</td>
          <td>22.258692</td>
          <td>0.005972</td>
          <td>22.324416</td>
          <td>0.005802</td>
          <td>23.488054</td>
          <td>0.014406</td>
          <td>25.744867</td>
          <td>0.192635</td>
          <td>25.422898</td>
          <td>0.314775</td>
          <td>23.737939</td>
          <td>20.384226</td>
          <td>22.815083</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.753537</td>
          <td>0.087319</td>
          <td>27.681906</td>
          <td>0.367843</td>
          <td>26.867078</td>
          <td>0.167602</td>
          <td>23.946306</td>
          <td>0.021044</td>
          <td>19.814546</td>
          <td>0.005116</td>
          <td>22.737282</td>
          <td>0.030929</td>
          <td>25.730922</td>
          <td>23.142357</td>
          <td>26.550749</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.859575</td>
          <td>0.008522</td>
          <td>22.219286</td>
          <td>0.005915</td>
          <td>20.977609</td>
          <td>0.005095</td>
          <td>20.233290</td>
          <td>0.005071</td>
          <td>21.852857</td>
          <td>0.007828</td>
          <td>26.458192</td>
          <td>0.680951</td>
          <td>11.442345</td>
          <td>20.401696</td>
          <td>28.636086</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.297655</td>
          <td>0.006636</td>
          <td>26.470775</td>
          <td>0.135045</td>
          <td>23.907867</td>
          <td>0.012979</td>
          <td>27.309042</td>
          <td>0.378543</td>
          <td>21.412526</td>
          <td>0.006457</td>
          <td>22.878675</td>
          <td>0.035032</td>
          <td>20.922308</td>
          <td>25.153080</td>
          <td>23.993964</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.443869</td>
          <td>0.007004</td>
          <td>24.384983</td>
          <td>0.021691</td>
          <td>18.989867</td>
          <td>0.005007</td>
          <td>24.970121</td>
          <td>0.051696</td>
          <td>22.195766</td>
          <td>0.009585</td>
          <td>22.572717</td>
          <td>0.026779</td>
          <td>22.904663</td>
          <td>23.185487</td>
          <td>21.214893</td>
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
          <td>27.597138</td>
          <td>23.410746</td>
          <td>24.332932</td>
          <td>21.738095</td>
          <td>25.762221</td>
          <td>22.901580</td>
          <td>22.859942</td>
          <td>0.006913</td>
          <td>24.170842</td>
          <td>0.027951</td>
          <td>21.215713</td>
          <td>0.005323</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.492572</td>
          <td>24.849699</td>
          <td>21.856477</td>
          <td>28.348929</td>
          <td>24.810721</td>
          <td>24.516700</td>
          <td>25.243721</td>
          <td>0.042524</td>
          <td>25.510416</td>
          <td>0.091837</td>
          <td>26.652818</td>
          <td>0.244428</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.528775</td>
          <td>24.695894</td>
          <td>19.489144</td>
          <td>22.534017</td>
          <td>22.105033</td>
          <td>28.391906</td>
          <td>27.673376</td>
          <td>0.343298</td>
          <td>23.528140</td>
          <td>0.016092</td>
          <td>23.053220</td>
          <td>0.011088</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.001564</td>
          <td>25.114046</td>
          <td>25.460844</td>
          <td>13.027881</td>
          <td>29.276571</td>
          <td>16.347359</td>
          <td>18.820148</td>
          <td>0.005001</td>
          <td>25.431686</td>
          <td>0.085678</td>
          <td>27.705607</td>
          <td>0.553997</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.741741</td>
          <td>19.055497</td>
          <td>20.419327</td>
          <td>21.201373</td>
          <td>19.648462</td>
          <td>22.229420</td>
          <td>20.823225</td>
          <td>0.005053</td>
          <td>24.411337</td>
          <td>0.034579</td>
          <td>23.087848</td>
          <td>0.011375</td>
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
          <td>20.784500</td>
          <td>22.244543</td>
          <td>22.330757</td>
          <td>23.481757</td>
          <td>25.612422</td>
          <td>25.472825</td>
          <td>23.718735</td>
          <td>0.011639</td>
          <td>20.387234</td>
          <td>0.005072</td>
          <td>22.819517</td>
          <td>0.009422</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.693647</td>
          <td>27.168555</td>
          <td>26.823348</td>
          <td>23.964394</td>
          <td>19.816125</td>
          <td>22.723520</td>
          <td>25.839393</td>
          <td>0.072270</td>
          <td>23.134508</td>
          <td>0.011778</td>
          <td>26.416626</td>
          <td>0.200789</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.858594</td>
          <td>22.220576</td>
          <td>20.974283</td>
          <td>20.229863</td>
          <td>21.850657</td>
          <td>26.088833</td>
          <td>11.443304</td>
          <td>0.005000</td>
          <td>20.400411</td>
          <td>0.005074</td>
          <td>28.217478</td>
          <td>0.788172</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.300828</td>
          <td>26.489568</td>
          <td>23.922031</td>
          <td>26.914451</td>
          <td>21.419059</td>
          <td>22.900724</td>
          <td>20.925271</td>
          <td>0.005064</td>
          <td>25.230300</td>
          <td>0.071689</td>
          <td>23.972169</td>
          <td>0.023486</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.449172</td>
          <td>24.369495</td>
          <td>18.990733</td>
          <td>24.942895</td>
          <td>22.184028</td>
          <td>22.599188</td>
          <td>22.900142</td>
          <td>0.007039</td>
          <td>23.200419</td>
          <td>0.012382</td>
          <td>21.219162</td>
          <td>0.005325</td>
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
          <td>27.597138</td>
          <td>23.410746</td>
          <td>24.332932</td>
          <td>21.738095</td>
          <td>25.762221</td>
          <td>22.901580</td>
          <td>22.855500</td>
          <td>0.056147</td>
          <td>24.108074</td>
          <td>0.141762</td>
          <td>21.225567</td>
          <td>0.012625</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.492572</td>
          <td>24.849699</td>
          <td>21.856477</td>
          <td>28.348929</td>
          <td>24.810721</td>
          <td>24.516700</td>
          <td>24.716920</td>
          <td>0.279554</td>
          <td>25.484755</td>
          <td>0.436889</td>
          <td>26.171335</td>
          <td>0.764601</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.528775</td>
          <td>24.695894</td>
          <td>19.489144</td>
          <td>22.534017</td>
          <td>22.105033</td>
          <td>28.391906</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.512650</td>
          <td>0.084249</td>
          <td>23.200052</td>
          <td>0.069790</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.001564</td>
          <td>25.114046</td>
          <td>25.460844</td>
          <td>13.027881</td>
          <td>29.276571</td>
          <td>16.347359</td>
          <td>18.825782</td>
          <td>0.005192</td>
          <td>24.969218</td>
          <td>0.291646</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.741741</td>
          <td>19.055497</td>
          <td>20.419327</td>
          <td>21.201373</td>
          <td>19.648462</td>
          <td>22.229420</td>
          <td>20.820811</td>
          <td>0.010090</td>
          <td>24.393330</td>
          <td>0.180941</td>
          <td>23.025741</td>
          <td>0.059771</td>
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
          <td>20.784500</td>
          <td>22.244543</td>
          <td>22.330757</td>
          <td>23.481757</td>
          <td>25.612422</td>
          <td>25.472825</td>
          <td>23.698714</td>
          <td>0.118230</td>
          <td>20.381447</td>
          <td>0.006980</td>
          <td>22.800650</td>
          <td>0.048909</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.693647</td>
          <td>27.168555</td>
          <td>26.823348</td>
          <td>23.964394</td>
          <td>19.816125</td>
          <td>22.723520</td>
          <td>25.294257</td>
          <td>0.440046</td>
          <td>23.145421</td>
          <td>0.060827</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.858594</td>
          <td>22.220576</td>
          <td>20.974283</td>
          <td>20.229863</td>
          <td>21.850657</td>
          <td>26.088833</td>
          <td>11.446687</td>
          <td>0.005000</td>
          <td>20.399652</td>
          <td>0.007037</td>
          <td>27.027379</td>
          <td>1.279362</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.300828</td>
          <td>26.489568</td>
          <td>23.922031</td>
          <td>26.914451</td>
          <td>21.419059</td>
          <td>22.900724</td>
          <td>20.899159</td>
          <td>0.010663</td>
          <td>24.773929</td>
          <td>0.248716</td>
          <td>23.755097</td>
          <td>0.113818</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.449172</td>
          <td>24.369495</td>
          <td>18.990733</td>
          <td>24.942895</td>
          <td>22.184028</td>
          <td>22.599188</td>
          <td>22.891534</td>
          <td>0.057978</td>
          <td>23.334772</td>
          <td>0.071974</td>
          <td>21.228030</td>
          <td>0.012649</td>
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

