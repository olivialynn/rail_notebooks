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
    /home/runner/.cache/lephare/runs/20260420T121550


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
      File "/tmp/ipykernel_5997/2313627096.py", line 5, in <module>
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
          <td>27.956795</td>
          <td>22.365219</td>
          <td>26.191840</td>
          <td>27.051298</td>
          <td>25.227091</td>
          <td>27.809676</td>
          <td>18.609407</td>
          <td>24.496708</td>
          <td>25.793000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.305055</td>
          <td>22.973526</td>
          <td>21.633102</td>
          <td>20.693843</td>
          <td>23.751262</td>
          <td>23.465623</td>
          <td>25.154047</td>
          <td>20.167565</td>
          <td>23.442214</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.617270</td>
          <td>23.856361</td>
          <td>26.072737</td>
          <td>24.255643</td>
          <td>15.089727</td>
          <td>22.989813</td>
          <td>25.651835</td>
          <td>24.586230</td>
          <td>25.994166</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.571273</td>
          <td>26.006368</td>
          <td>26.506783</td>
          <td>25.036668</td>
          <td>25.928418</td>
          <td>18.144953</td>
          <td>22.349941</td>
          <td>27.171763</td>
          <td>17.642735</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.973053</td>
          <td>25.065905</td>
          <td>26.714899</td>
          <td>23.652797</td>
          <td>21.455184</td>
          <td>19.579376</td>
          <td>26.112414</td>
          <td>21.125654</td>
          <td>24.762311</td>
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
          <td>20.290490</td>
          <td>24.416668</td>
          <td>32.344586</td>
          <td>26.857999</td>
          <td>23.385406</td>
          <td>22.790256</td>
          <td>24.041754</td>
          <td>22.329493</td>
          <td>21.462117</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.812178</td>
          <td>24.546504</td>
          <td>20.266806</td>
          <td>18.772264</td>
          <td>26.305744</td>
          <td>21.579055</td>
          <td>24.354949</td>
          <td>22.210647</td>
          <td>26.551380</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.500222</td>
          <td>17.222725</td>
          <td>24.903217</td>
          <td>23.928562</td>
          <td>23.384174</td>
          <td>24.845302</td>
          <td>26.518030</td>
          <td>21.732076</td>
          <td>24.987381</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.835272</td>
          <td>19.321233</td>
          <td>25.307414</td>
          <td>22.434035</td>
          <td>16.609053</td>
          <td>20.995181</td>
          <td>21.929058</td>
          <td>25.332905</td>
          <td>23.371885</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.124210</td>
          <td>20.627478</td>
          <td>23.961882</td>
          <td>28.589283</td>
          <td>19.734946</td>
          <td>21.863868</td>
          <td>23.840550</td>
          <td>27.708986</td>
          <td>23.522168</td>
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
          <td>28.209823</td>
          <td>1.193756</td>
          <td>22.371757</td>
          <td>0.006156</td>
          <td>26.313843</td>
          <td>0.103898</td>
          <td>26.937851</td>
          <td>0.281874</td>
          <td>25.298318</td>
          <td>0.131529</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.609407</td>
          <td>24.496708</td>
          <td>25.793000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.428745</td>
          <td>0.065655</td>
          <td>22.977243</td>
          <td>0.007840</td>
          <td>21.632689</td>
          <td>0.005263</td>
          <td>20.694094</td>
          <td>0.005143</td>
          <td>23.763604</td>
          <td>0.033985</td>
          <td>23.415604</td>
          <td>0.056388</td>
          <td>25.154047</td>
          <td>20.167565</td>
          <td>23.442214</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.613478</td>
          <td>0.005054</td>
          <td>23.843768</td>
          <td>0.013922</td>
          <td>26.096569</td>
          <td>0.085855</td>
          <td>24.246136</td>
          <td>0.027268</td>
          <td>15.090823</td>
          <td>0.005001</td>
          <td>22.986874</td>
          <td>0.038549</td>
          <td>25.651835</td>
          <td>24.586230</td>
          <td>25.994166</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.707685</td>
          <td>0.083881</td>
          <td>26.017903</td>
          <td>0.091022</td>
          <td>26.607423</td>
          <td>0.134123</td>
          <td>24.977036</td>
          <td>0.052015</td>
          <td>25.979734</td>
          <td>0.234385</td>
          <td>18.146579</td>
          <td>0.005039</td>
          <td>22.349941</td>
          <td>27.171763</td>
          <td>17.642735</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.851797</td>
          <td>0.095154</td>
          <td>25.099698</td>
          <td>0.040433</td>
          <td>26.888207</td>
          <td>0.170644</td>
          <td>23.633943</td>
          <td>0.016204</td>
          <td>21.451538</td>
          <td>0.006547</td>
          <td>19.585026</td>
          <td>0.005343</td>
          <td>26.112414</td>
          <td>21.125654</td>
          <td>24.762311</td>
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
          <td>20.297596</td>
          <td>0.005410</td>
          <td>24.392586</td>
          <td>0.021832</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.903471</td>
          <td>0.274115</td>
          <td>23.417185</td>
          <td>0.025091</td>
          <td>22.751022</td>
          <td>0.031305</td>
          <td>24.041754</td>
          <td>22.329493</td>
          <td>21.462117</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.816624</td>
          <td>0.005835</td>
          <td>24.501371</td>
          <td>0.023963</td>
          <td>20.268213</td>
          <td>0.005034</td>
          <td>18.767288</td>
          <td>0.005010</td>
          <td>26.002459</td>
          <td>0.238830</td>
          <td>21.565799</td>
          <td>0.011723</td>
          <td>24.354949</td>
          <td>22.210647</td>
          <td>26.551380</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.504348</td>
          <td>0.007179</td>
          <td>17.228300</td>
          <td>0.005002</td>
          <td>24.846627</td>
          <td>0.028413</td>
          <td>23.927199</td>
          <td>0.020704</td>
          <td>23.387110</td>
          <td>0.024446</td>
          <td>24.832330</td>
          <td>0.193728</td>
          <td>26.518030</td>
          <td>21.732076</td>
          <td>24.987381</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.868893</td>
          <td>0.226948</td>
          <td>19.318787</td>
          <td>0.005017</td>
          <td>25.340411</td>
          <td>0.043935</td>
          <td>22.426700</td>
          <td>0.007228</td>
          <td>16.602896</td>
          <td>0.005002</td>
          <td>20.987856</td>
          <td>0.008061</td>
          <td>21.929058</td>
          <td>25.332905</td>
          <td>23.371885</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.117296</td>
          <td>0.021045</td>
          <td>20.629714</td>
          <td>0.005086</td>
          <td>23.945063</td>
          <td>0.013357</td>
          <td>28.930041</td>
          <td>1.139304</td>
          <td>19.734228</td>
          <td>0.005103</td>
          <td>21.853111</td>
          <td>0.014617</td>
          <td>23.840550</td>
          <td>27.708986</td>
          <td>23.522168</td>
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
          <td>27.956795</td>
          <td>22.365219</td>
          <td>26.191840</td>
          <td>27.051298</td>
          <td>25.227091</td>
          <td>27.809676</td>
          <td>18.616875</td>
          <td>0.005001</td>
          <td>24.418563</td>
          <td>0.034802</td>
          <td>26.033006</td>
          <td>0.144842</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.305055</td>
          <td>22.973526</td>
          <td>21.633102</td>
          <td>20.693843</td>
          <td>23.751262</td>
          <td>23.465623</td>
          <td>25.236053</td>
          <td>0.042234</td>
          <td>20.180867</td>
          <td>0.005049</td>
          <td>23.432454</td>
          <td>0.014878</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.617270</td>
          <td>23.856361</td>
          <td>26.072737</td>
          <td>24.255643</td>
          <td>15.089727</td>
          <td>22.989813</td>
          <td>25.547889</td>
          <td>0.055767</td>
          <td>24.636032</td>
          <td>0.042233</td>
          <td>26.169707</td>
          <td>0.162867</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.571273</td>
          <td>26.006368</td>
          <td>26.506783</td>
          <td>25.036668</td>
          <td>25.928418</td>
          <td>18.144953</td>
          <td>22.348464</td>
          <td>0.005822</td>
          <td>26.510188</td>
          <td>0.217155</td>
          <td>17.643010</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.973053</td>
          <td>25.065905</td>
          <td>26.714899</td>
          <td>23.652797</td>
          <td>21.455184</td>
          <td>19.579376</td>
          <td>26.132535</td>
          <td>0.093643</td>
          <td>21.134519</td>
          <td>0.005279</td>
          <td>24.742064</td>
          <td>0.046420</td>
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
          <td>20.290490</td>
          <td>24.416668</td>
          <td>32.344586</td>
          <td>26.857999</td>
          <td>23.385406</td>
          <td>22.790256</td>
          <td>24.016415</td>
          <td>0.014685</td>
          <td>22.323083</td>
          <td>0.007114</td>
          <td>21.469278</td>
          <td>0.005506</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.812178</td>
          <td>24.546504</td>
          <td>20.266806</td>
          <td>18.772264</td>
          <td>26.305744</td>
          <td>21.579055</td>
          <td>24.368050</td>
          <td>0.019692</td>
          <td>22.218055</td>
          <td>0.006790</td>
          <td>26.591760</td>
          <td>0.232397</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.500222</td>
          <td>17.222725</td>
          <td>24.903217</td>
          <td>23.928562</td>
          <td>23.384174</td>
          <td>24.845302</td>
          <td>26.552664</td>
          <td>0.135137</td>
          <td>21.736030</td>
          <td>0.005804</td>
          <td>24.932404</td>
          <td>0.055003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.835272</td>
          <td>19.321233</td>
          <td>25.307414</td>
          <td>22.434035</td>
          <td>16.609053</td>
          <td>20.995181</td>
          <td>21.936832</td>
          <td>0.005401</td>
          <td>25.315022</td>
          <td>0.077280</td>
          <td>23.365522</td>
          <td>0.014096</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.124210</td>
          <td>20.627478</td>
          <td>23.961882</td>
          <td>28.589283</td>
          <td>19.734946</td>
          <td>21.863868</td>
          <td>23.851903</td>
          <td>0.012885</td>
          <td>29.283270</td>
          <td>1.462715</td>
          <td>23.535288</td>
          <td>0.016187</td>
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
          <td>27.956795</td>
          <td>22.365219</td>
          <td>26.191840</td>
          <td>27.051298</td>
          <td>25.227091</td>
          <td>27.809676</td>
          <td>18.610152</td>
          <td>0.005130</td>
          <td>24.289085</td>
          <td>0.165585</td>
          <td>25.670771</td>
          <td>0.540209</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.305055</td>
          <td>22.973526</td>
          <td>21.633102</td>
          <td>20.693843</td>
          <td>23.751262</td>
          <td>23.465623</td>
          <td>25.189697</td>
          <td>0.406314</td>
          <td>20.182827</td>
          <td>0.006439</td>
          <td>23.341735</td>
          <td>0.079129</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.617270</td>
          <td>23.856361</td>
          <td>26.072737</td>
          <td>24.255643</td>
          <td>15.089727</td>
          <td>22.989813</td>
          <td>24.637957</td>
          <td>0.262131</td>
          <td>24.937851</td>
          <td>0.284340</td>
          <td>27.807341</td>
          <td>1.872953</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.571273</td>
          <td>26.006368</td>
          <td>26.506783</td>
          <td>25.036668</td>
          <td>25.928418</td>
          <td>18.144953</td>
          <td>22.403801</td>
          <td>0.037542</td>
          <td>27.316989</td>
          <td>1.414058</td>
          <td>17.644966</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.973053</td>
          <td>25.065905</td>
          <td>26.714899</td>
          <td>23.652797</td>
          <td>21.455184</td>
          <td>19.579376</td>
          <td>26.953136</td>
          <td>1.297256</td>
          <td>21.112128</td>
          <td>0.010763</td>
          <td>24.794063</td>
          <td>0.274409</td>
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
          <td>20.290490</td>
          <td>24.416668</td>
          <td>32.344586</td>
          <td>26.857999</td>
          <td>23.385406</td>
          <td>22.790256</td>
          <td>23.980732</td>
          <td>0.150912</td>
          <td>22.341695</td>
          <td>0.029754</td>
          <td>21.465256</td>
          <td>0.015281</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.812178</td>
          <td>24.546504</td>
          <td>20.266806</td>
          <td>18.772264</td>
          <td>26.305744</td>
          <td>21.579055</td>
          <td>24.638033</td>
          <td>0.262147</td>
          <td>22.181091</td>
          <td>0.025831</td>
          <td>26.125876</td>
          <td>0.741860</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.500222</td>
          <td>17.222725</td>
          <td>24.903217</td>
          <td>23.928562</td>
          <td>23.384174</td>
          <td>24.845302</td>
          <td>25.875213</td>
          <td>0.669887</td>
          <td>21.721185</td>
          <td>0.017389</td>
          <td>25.902063</td>
          <td>0.636818</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.835272</td>
          <td>19.321233</td>
          <td>25.307414</td>
          <td>22.434035</td>
          <td>16.609053</td>
          <td>20.995181</td>
          <td>21.930246</td>
          <td>0.024707</td>
          <td>24.982881</td>
          <td>0.294879</td>
          <td>23.316830</td>
          <td>0.077404</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.124210</td>
          <td>20.627478</td>
          <td>23.961882</td>
          <td>28.589283</td>
          <td>19.734946</td>
          <td>21.863868</td>
          <td>23.686312</td>
          <td>0.116959</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.621266</td>
          <td>0.101237</td>
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

