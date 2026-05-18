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
    /home/runner/.cache/lephare/runs/20260518T131208


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
      File "/tmp/ipykernel_5617/2313627096.py", line 5, in <module>
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
          <td>21.117139</td>
          <td>20.283860</td>
          <td>25.086121</td>
          <td>18.494563</td>
          <td>19.179709</td>
          <td>23.578716</td>
          <td>22.768172</td>
          <td>26.231203</td>
          <td>26.716186</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.302199</td>
          <td>18.705203</td>
          <td>21.497020</td>
          <td>25.966725</td>
          <td>26.721291</td>
          <td>22.739537</td>
          <td>25.163256</td>
          <td>28.819317</td>
          <td>27.879847</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.900034</td>
          <td>20.839011</td>
          <td>22.789610</td>
          <td>22.317838</td>
          <td>24.813508</td>
          <td>18.340869</td>
          <td>30.972153</td>
          <td>27.621852</td>
          <td>23.720431</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759048</td>
          <td>20.682038</td>
          <td>18.966852</td>
          <td>20.769918</td>
          <td>24.064903</td>
          <td>21.789298</td>
          <td>22.878454</td>
          <td>22.235230</td>
          <td>21.945743</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.737193</td>
          <td>22.023999</td>
          <td>23.335590</td>
          <td>26.199400</td>
          <td>20.418947</td>
          <td>23.362594</td>
          <td>22.110785</td>
          <td>19.547389</td>
          <td>26.408974</td>
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
          <td>26.412678</td>
          <td>22.103901</td>
          <td>25.388719</td>
          <td>23.086252</td>
          <td>20.720952</td>
          <td>23.710101</td>
          <td>28.362199</td>
          <td>21.283402</td>
          <td>27.046869</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.735036</td>
          <td>22.680640</td>
          <td>19.356330</td>
          <td>21.239719</td>
          <td>28.339540</td>
          <td>21.156154</td>
          <td>23.980793</td>
          <td>20.849292</td>
          <td>22.345674</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.825164</td>
          <td>26.089293</td>
          <td>29.294543</td>
          <td>25.395346</td>
          <td>25.202285</td>
          <td>25.266750</td>
          <td>16.888739</td>
          <td>23.830501</td>
          <td>19.140619</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.353061</td>
          <td>23.652777</td>
          <td>22.437126</td>
          <td>21.431780</td>
          <td>18.426674</td>
          <td>26.352598</td>
          <td>19.950886</td>
          <td>23.777785</td>
          <td>22.791791</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.815082</td>
          <td>16.328094</td>
          <td>23.271905</td>
          <td>16.579906</td>
          <td>23.381658</td>
          <td>20.634801</td>
          <td>20.953419</td>
          <td>21.633546</td>
          <td>23.739866</td>
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
          <td>21.118219</td>
          <td>0.006273</td>
          <td>20.287011</td>
          <td>0.005055</td>
          <td>25.149212</td>
          <td>0.037088</td>
          <td>18.490712</td>
          <td>0.005007</td>
          <td>19.184127</td>
          <td>0.005046</td>
          <td>23.496653</td>
          <td>0.060593</td>
          <td>22.768172</td>
          <td>26.231203</td>
          <td>26.716186</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.305051</td>
          <td>0.005117</td>
          <td>18.699173</td>
          <td>0.005009</td>
          <td>21.487189</td>
          <td>0.005209</td>
          <td>25.874979</td>
          <td>0.114841</td>
          <td>27.485886</td>
          <td>0.732816</td>
          <td>22.775019</td>
          <td>0.031972</td>
          <td>25.163256</td>
          <td>28.819317</td>
          <td>27.879847</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.907905</td>
          <td>0.005949</td>
          <td>20.832583</td>
          <td>0.005114</td>
          <td>22.778593</td>
          <td>0.006642</td>
          <td>22.311833</td>
          <td>0.006873</td>
          <td>24.857753</td>
          <td>0.089539</td>
          <td>18.344705</td>
          <td>0.005052</td>
          <td>30.972153</td>
          <td>27.621852</td>
          <td>23.720431</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.755495</td>
          <td>0.005767</td>
          <td>20.681898</td>
          <td>0.005093</td>
          <td>18.969601</td>
          <td>0.005007</td>
          <td>20.758940</td>
          <td>0.005158</td>
          <td>24.012126</td>
          <td>0.042343</td>
          <td>21.780792</td>
          <td>0.013806</td>
          <td>22.878454</td>
          <td>22.235230</td>
          <td>21.945743</td>
        </tr>
        <tr>
          <th>4</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.023065</td>
          <td>0.005676</td>
          <td>23.335729</td>
          <td>0.008738</td>
          <td>26.353234</td>
          <td>0.173365</td>
          <td>20.415944</td>
          <td>0.005297</td>
          <td>23.398298</td>
          <td>0.055529</td>
          <td>22.110785</td>
          <td>19.547389</td>
          <td>26.408974</td>
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
          <td>26.785604</td>
          <td>0.468727</td>
          <td>22.095694</td>
          <td>0.005757</td>
          <td>25.379523</td>
          <td>0.045487</td>
          <td>23.095320</td>
          <td>0.010721</td>
          <td>20.713784</td>
          <td>0.005479</td>
          <td>23.679324</td>
          <td>0.071237</td>
          <td>28.362199</td>
          <td>21.283402</td>
          <td>27.046869</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.739345</td>
          <td>0.005022</td>
          <td>22.679577</td>
          <td>0.006840</td>
          <td>19.359051</td>
          <td>0.005010</td>
          <td>21.241933</td>
          <td>0.005341</td>
          <td>28.570417</td>
          <td>1.395862</td>
          <td>21.154554</td>
          <td>0.008877</td>
          <td>23.980793</td>
          <td>20.849292</td>
          <td>22.345674</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.752293</td>
          <td>0.087224</td>
          <td>25.944143</td>
          <td>0.085309</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.389506</td>
          <td>0.074978</td>
          <td>25.086628</td>
          <td>0.109426</td>
          <td>24.950483</td>
          <td>0.213906</td>
          <td>16.888739</td>
          <td>23.830501</td>
          <td>19.140619</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.357042</td>
          <td>0.006777</td>
          <td>23.667442</td>
          <td>0.012176</td>
          <td>22.444178</td>
          <td>0.005971</td>
          <td>21.434752</td>
          <td>0.005465</td>
          <td>18.429449</td>
          <td>0.005017</td>
          <td>27.063925</td>
          <td>1.005131</td>
          <td>19.950886</td>
          <td>23.777785</td>
          <td>22.791791</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.753107</td>
          <td>0.087286</td>
          <td>16.335632</td>
          <td>0.005001</td>
          <td>23.266642</td>
          <td>0.008390</td>
          <td>16.574244</td>
          <td>0.005001</td>
          <td>23.390484</td>
          <td>0.024517</td>
          <td>20.633805</td>
          <td>0.006809</td>
          <td>20.953419</td>
          <td>21.633546</td>
          <td>23.739866</td>
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
          <td>21.117139</td>
          <td>20.283860</td>
          <td>25.086121</td>
          <td>18.494563</td>
          <td>19.179709</td>
          <td>23.578716</td>
          <td>22.775346</td>
          <td>0.006672</td>
          <td>26.625945</td>
          <td>0.239066</td>
          <td>27.426327</td>
          <td>0.450837</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.302199</td>
          <td>18.705203</td>
          <td>21.497020</td>
          <td>25.966725</td>
          <td>26.721291</td>
          <td>22.739537</td>
          <td>25.133508</td>
          <td>0.038548</td>
          <td>30.722290</td>
          <td>2.671750</td>
          <td>27.864513</td>
          <td>0.620317</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.900034</td>
          <td>20.839011</td>
          <td>22.789610</td>
          <td>22.317838</td>
          <td>24.813508</td>
          <td>18.340869</td>
          <td>29.024007</td>
          <td>0.899653</td>
          <td>27.888763</td>
          <td>0.630937</td>
          <td>23.712683</td>
          <td>0.018785</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759048</td>
          <td>20.682038</td>
          <td>18.966852</td>
          <td>20.769918</td>
          <td>24.064903</td>
          <td>21.789298</td>
          <td>22.875142</td>
          <td>0.006960</td>
          <td>22.234789</td>
          <td>0.006838</td>
          <td>21.947717</td>
          <td>0.006151</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.737193</td>
          <td>22.023999</td>
          <td>23.335590</td>
          <td>26.199400</td>
          <td>20.418947</td>
          <td>23.362594</td>
          <td>22.111511</td>
          <td>0.005545</td>
          <td>19.555879</td>
          <td>0.005016</td>
          <td>26.619346</td>
          <td>0.237766</td>
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
          <td>26.412678</td>
          <td>22.103901</td>
          <td>25.388719</td>
          <td>23.086252</td>
          <td>20.720952</td>
          <td>23.710101</td>
          <td>28.149142</td>
          <td>0.494145</td>
          <td>21.286844</td>
          <td>0.005367</td>
          <td>27.646692</td>
          <td>0.530834</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.735036</td>
          <td>22.680640</td>
          <td>19.356330</td>
          <td>21.239719</td>
          <td>28.339540</td>
          <td>21.156154</td>
          <td>23.976824</td>
          <td>0.014224</td>
          <td>20.846506</td>
          <td>0.005166</td>
          <td>22.340242</td>
          <td>0.007171</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.825164</td>
          <td>26.089293</td>
          <td>29.294543</td>
          <td>25.395346</td>
          <td>25.202285</td>
          <td>25.266750</td>
          <td>16.889584</td>
          <td>0.005000</td>
          <td>23.844018</td>
          <td>0.021019</td>
          <td>19.146258</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.353061</td>
          <td>23.652777</td>
          <td>22.437126</td>
          <td>21.431780</td>
          <td>18.426674</td>
          <td>26.352598</td>
          <td>19.955522</td>
          <td>0.005011</td>
          <td>23.836206</td>
          <td>0.020878</td>
          <td>22.805791</td>
          <td>0.009337</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.815082</td>
          <td>16.328094</td>
          <td>23.271905</td>
          <td>16.579906</td>
          <td>23.381658</td>
          <td>20.634801</td>
          <td>20.956805</td>
          <td>0.005068</td>
          <td>21.626324</td>
          <td>0.005666</td>
          <td>23.697133</td>
          <td>0.018539</td>
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
          <td>21.117139</td>
          <td>20.283860</td>
          <td>25.086121</td>
          <td>18.494563</td>
          <td>19.179709</td>
          <td>23.578716</td>
          <td>22.786758</td>
          <td>0.052811</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.302199</td>
          <td>18.705203</td>
          <td>21.497020</td>
          <td>25.966725</td>
          <td>26.721291</td>
          <td>22.739537</td>
          <td>25.075222</td>
          <td>0.371859</td>
          <td>26.391506</td>
          <td>0.827009</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.900034</td>
          <td>20.839011</td>
          <td>22.789610</td>
          <td>22.317838</td>
          <td>24.813508</td>
          <td>18.340869</td>
          <td>27.577922</td>
          <td>1.767750</td>
          <td>25.869797</td>
          <td>0.580106</td>
          <td>23.677060</td>
          <td>0.106311</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759048</td>
          <td>20.682038</td>
          <td>18.966852</td>
          <td>20.769918</td>
          <td>24.064903</td>
          <td>21.789298</td>
          <td>22.858613</td>
          <td>0.056303</td>
          <td>22.224386</td>
          <td>0.026832</td>
          <td>21.948536</td>
          <td>0.023008</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.737193</td>
          <td>22.023999</td>
          <td>23.335590</td>
          <td>26.199400</td>
          <td>20.418947</td>
          <td>23.362594</td>
          <td>22.117541</td>
          <td>0.029126</td>
          <td>19.545884</td>
          <td>0.005486</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>26.412678</td>
          <td>22.103901</td>
          <td>25.388719</td>
          <td>23.086252</td>
          <td>20.720952</td>
          <td>23.710101</td>
          <td>27.508750</td>
          <td>1.712508</td>
          <td>21.269497</td>
          <td>0.012093</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.735036</td>
          <td>22.680640</td>
          <td>19.356330</td>
          <td>21.239719</td>
          <td>28.339540</td>
          <td>21.156154</td>
          <td>23.681325</td>
          <td>0.116452</td>
          <td>20.854156</td>
          <td>0.009031</td>
          <td>22.365784</td>
          <td>0.033209</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.825164</td>
          <td>26.089293</td>
          <td>29.294543</td>
          <td>25.395346</td>
          <td>25.202285</td>
          <td>25.266750</td>
          <td>16.881968</td>
          <td>0.005005</td>
          <td>23.800649</td>
          <td>0.108529</td>
          <td>19.132052</td>
          <td>0.005278</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.353061</td>
          <td>23.652777</td>
          <td>22.437126</td>
          <td>21.431780</td>
          <td>18.426674</td>
          <td>26.352598</td>
          <td>19.936785</td>
          <td>0.006334</td>
          <td>23.625214</td>
          <td>0.093042</td>
          <td>22.837448</td>
          <td>0.050540</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.815082</td>
          <td>16.328094</td>
          <td>23.271905</td>
          <td>16.579906</td>
          <td>23.381658</td>
          <td>20.634801</td>
          <td>20.938715</td>
          <td>0.010972</td>
          <td>21.639132</td>
          <td>0.016239</td>
          <td>23.684033</td>
          <td>0.106962</td>
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

