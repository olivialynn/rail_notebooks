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
    /home/runner/.cache/lephare/runs/20260406T120756


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
      File "/tmp/ipykernel_4409/2313627096.py", line 5, in <module>
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
          <td>18.152004</td>
          <td>14.839531</td>
          <td>25.818610</td>
          <td>19.165352</td>
          <td>22.720172</td>
          <td>24.565065</td>
          <td>20.969880</td>
          <td>19.022509</td>
          <td>28.505730</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.878617</td>
          <td>24.715747</td>
          <td>18.689277</td>
          <td>22.316884</td>
          <td>20.962630</td>
          <td>22.940058</td>
          <td>22.509761</td>
          <td>17.335609</td>
          <td>27.593460</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.684530</td>
          <td>25.017599</td>
          <td>18.473462</td>
          <td>23.995595</td>
          <td>25.115175</td>
          <td>19.536624</td>
          <td>25.607380</td>
          <td>22.120377</td>
          <td>21.846070</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.448573</td>
          <td>18.117164</td>
          <td>22.750942</td>
          <td>27.251538</td>
          <td>25.826447</td>
          <td>16.319795</td>
          <td>25.673589</td>
          <td>19.644903</td>
          <td>23.518194</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.981701</td>
          <td>18.755103</td>
          <td>25.957058</td>
          <td>26.607032</td>
          <td>26.290212</td>
          <td>27.807966</td>
          <td>26.957054</td>
          <td>21.623847</td>
          <td>27.274703</td>
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
          <td>22.823508</td>
          <td>23.643726</td>
          <td>27.155182</td>
          <td>26.115826</td>
          <td>21.502638</td>
          <td>28.778459</td>
          <td>16.447117</td>
          <td>22.029002</td>
          <td>23.558138</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.624859</td>
          <td>25.264117</td>
          <td>22.450229</td>
          <td>21.450214</td>
          <td>22.630703</td>
          <td>26.118446</td>
          <td>21.745600</td>
          <td>30.930371</td>
          <td>21.201984</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.505806</td>
          <td>25.450549</td>
          <td>27.564398</td>
          <td>22.949995</td>
          <td>20.536780</td>
          <td>22.976313</td>
          <td>21.441748</td>
          <td>22.182572</td>
          <td>21.629989</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178153</td>
          <td>25.263392</td>
          <td>24.644918</td>
          <td>19.303609</td>
          <td>21.526562</td>
          <td>25.128546</td>
          <td>26.197657</td>
          <td>19.773765</td>
          <td>19.453594</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.340817</td>
          <td>26.562563</td>
          <td>26.636875</td>
          <td>21.263552</td>
          <td>23.461059</td>
          <td>28.890656</td>
          <td>26.203382</td>
          <td>23.542898</td>
          <td>24.147256</td>
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
          <td>18.151587</td>
          <td>0.005033</td>
          <td>14.838745</td>
          <td>0.005000</td>
          <td>25.815761</td>
          <td>0.066991</td>
          <td>19.171529</td>
          <td>0.005016</td>
          <td>22.747469</td>
          <td>0.014329</td>
          <td>24.647354</td>
          <td>0.165619</td>
          <td>20.969880</td>
          <td>19.022509</td>
          <td>28.505730</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.870916</td>
          <td>0.040209</td>
          <td>24.746401</td>
          <td>0.029636</td>
          <td>18.694013</td>
          <td>0.005005</td>
          <td>22.320832</td>
          <td>0.006899</td>
          <td>20.953646</td>
          <td>0.005705</td>
          <td>22.884761</td>
          <td>0.035221</td>
          <td>22.509761</td>
          <td>17.335609</td>
          <td>27.593460</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.807071</td>
          <td>0.476292</td>
          <td>24.954916</td>
          <td>0.035585</td>
          <td>18.476135</td>
          <td>0.005004</td>
          <td>23.957923</td>
          <td>0.021254</td>
          <td>25.154918</td>
          <td>0.116138</td>
          <td>19.534427</td>
          <td>0.005316</td>
          <td>25.607380</td>
          <td>22.120377</td>
          <td>21.846070</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.444701</td>
          <td>0.005138</td>
          <td>18.123772</td>
          <td>0.005005</td>
          <td>22.758401</td>
          <td>0.006591</td>
          <td>27.429465</td>
          <td>0.415378</td>
          <td>26.280491</td>
          <td>0.299597</td>
          <td>16.320435</td>
          <td>0.005004</td>
          <td>25.673589</td>
          <td>19.644903</td>
          <td>23.518194</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.974903</td>
          <td>0.009098</td>
          <td>18.761712</td>
          <td>0.005009</td>
          <td>26.025980</td>
          <td>0.080675</td>
          <td>27.075074</td>
          <td>0.314787</td>
          <td>26.183605</td>
          <td>0.277032</td>
          <td>27.285152</td>
          <td>1.143991</td>
          <td>26.957054</td>
          <td>21.623847</td>
          <td>27.274703</td>
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
          <td>22.789685</td>
          <td>0.016120</td>
          <td>23.639203</td>
          <td>0.011925</td>
          <td>27.273109</td>
          <td>0.235725</td>
          <td>26.008904</td>
          <td>0.129008</td>
          <td>21.500052</td>
          <td>0.006667</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.447117</td>
          <td>22.029002</td>
          <td>23.558138</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.419874</td>
          <td>0.155507</td>
          <td>25.235983</td>
          <td>0.045612</td>
          <td>22.455079</td>
          <td>0.005988</td>
          <td>21.447464</td>
          <td>0.005475</td>
          <td>22.609605</td>
          <td>0.012871</td>
          <td>26.164591</td>
          <td>0.553985</td>
          <td>21.745600</td>
          <td>30.930371</td>
          <td>21.201984</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.506271</td>
          <td>0.005048</td>
          <td>25.428563</td>
          <td>0.054094</td>
          <td>27.431540</td>
          <td>0.268478</td>
          <td>22.955820</td>
          <td>0.009747</td>
          <td>20.535859</td>
          <td>0.005360</td>
          <td>22.994022</td>
          <td>0.038793</td>
          <td>21.441748</td>
          <td>22.182572</td>
          <td>21.629989</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.179198</td>
          <td>0.005350</td>
          <td>25.303266</td>
          <td>0.048412</td>
          <td>24.600887</td>
          <td>0.022951</td>
          <td>19.306136</td>
          <td>0.005019</td>
          <td>21.529282</td>
          <td>0.006744</td>
          <td>25.780495</td>
          <td>0.416379</td>
          <td>26.197657</td>
          <td>19.773765</td>
          <td>19.453594</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.353842</td>
          <td>0.006769</td>
          <td>26.543566</td>
          <td>0.143784</td>
          <td>26.919510</td>
          <td>0.175245</td>
          <td>21.258316</td>
          <td>0.005350</td>
          <td>23.454281</td>
          <td>0.025913</td>
          <td>27.085940</td>
          <td>1.018474</td>
          <td>26.203382</td>
          <td>23.542898</td>
          <td>24.147256</td>
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
          <td>18.152004</td>
          <td>14.839531</td>
          <td>25.818610</td>
          <td>19.165352</td>
          <td>22.720172</td>
          <td>24.565065</td>
          <td>20.971493</td>
          <td>0.005070</td>
          <td>19.025573</td>
          <td>0.005006</td>
          <td>27.446747</td>
          <td>0.457819</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.878617</td>
          <td>24.715747</td>
          <td>18.689277</td>
          <td>22.316884</td>
          <td>20.962630</td>
          <td>22.940058</td>
          <td>22.515285</td>
          <td>0.006090</td>
          <td>17.337781</td>
          <td>0.005000</td>
          <td>27.211709</td>
          <td>0.382566</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.684530</td>
          <td>25.017599</td>
          <td>18.473462</td>
          <td>23.995595</td>
          <td>25.115175</td>
          <td>19.536624</td>
          <td>25.686585</td>
          <td>0.063096</td>
          <td>22.109721</td>
          <td>0.006503</td>
          <td>21.841837</td>
          <td>0.005963</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.448573</td>
          <td>18.117164</td>
          <td>22.750942</td>
          <td>27.251538</td>
          <td>25.826447</td>
          <td>16.319795</td>
          <td>25.662863</td>
          <td>0.061779</td>
          <td>19.642470</td>
          <td>0.005018</td>
          <td>23.527890</td>
          <td>0.016088</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.981701</td>
          <td>18.755103</td>
          <td>25.957058</td>
          <td>26.607032</td>
          <td>26.290212</td>
          <td>27.807966</td>
          <td>27.203955</td>
          <td>0.234757</td>
          <td>21.621857</td>
          <td>0.005661</td>
          <td>27.339069</td>
          <td>0.421968</td>
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
          <td>22.823508</td>
          <td>23.643726</td>
          <td>27.155182</td>
          <td>26.115826</td>
          <td>21.502638</td>
          <td>28.778459</td>
          <td>16.443662</td>
          <td>0.005000</td>
          <td>22.016354</td>
          <td>0.006290</td>
          <td>23.570822</td>
          <td>0.016672</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.624859</td>
          <td>25.264117</td>
          <td>22.450229</td>
          <td>21.450214</td>
          <td>22.630703</td>
          <td>26.118446</td>
          <td>21.746860</td>
          <td>0.005286</td>
          <td>29.362660</td>
          <td>1.522018</td>
          <td>21.203369</td>
          <td>0.005316</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.505806</td>
          <td>25.450549</td>
          <td>27.564398</td>
          <td>22.949995</td>
          <td>20.536780</td>
          <td>22.976313</td>
          <td>21.444336</td>
          <td>0.005166</td>
          <td>22.186352</td>
          <td>0.006701</td>
          <td>21.631902</td>
          <td>0.005672</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178153</td>
          <td>25.263392</td>
          <td>24.644918</td>
          <td>19.303609</td>
          <td>21.526562</td>
          <td>25.128546</td>
          <td>25.918083</td>
          <td>0.077490</td>
          <td>19.775767</td>
          <td>0.005023</td>
          <td>19.457273</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.340817</td>
          <td>26.562563</td>
          <td>26.636875</td>
          <td>21.263552</td>
          <td>23.461059</td>
          <td>28.890656</td>
          <td>26.288179</td>
          <td>0.107351</td>
          <td>23.541606</td>
          <td>0.016272</td>
          <td>24.161483</td>
          <td>0.027722</td>
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
          <td>18.152004</td>
          <td>14.839531</td>
          <td>25.818610</td>
          <td>19.165352</td>
          <td>22.720172</td>
          <td>24.565065</td>
          <td>20.941894</td>
          <td>0.010997</td>
          <td>19.021392</td>
          <td>0.005190</td>
          <td>27.527573</td>
          <td>1.648602</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.878617</td>
          <td>24.715747</td>
          <td>18.689277</td>
          <td>22.316884</td>
          <td>20.962630</td>
          <td>22.940058</td>
          <td>22.458842</td>
          <td>0.039427</td>
          <td>17.332057</td>
          <td>0.005009</td>
          <td>26.359066</td>
          <td>0.863549</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.684530</td>
          <td>25.017599</td>
          <td>18.473462</td>
          <td>23.995595</td>
          <td>25.115175</td>
          <td>19.536624</td>
          <td>24.914231</td>
          <td>0.327583</td>
          <td>22.148081</td>
          <td>0.025095</td>
          <td>21.810138</td>
          <td>0.020415</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.448573</td>
          <td>18.117164</td>
          <td>22.750942</td>
          <td>27.251538</td>
          <td>25.826447</td>
          <td>16.319795</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.649620</td>
          <td>0.005583</td>
          <td>23.381835</td>
          <td>0.081986</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.981701</td>
          <td>18.755103</td>
          <td>25.957058</td>
          <td>26.607032</td>
          <td>26.290212</td>
          <td>27.807966</td>
          <td>26.087783</td>
          <td>0.772947</td>
          <td>21.645477</td>
          <td>0.016324</td>
          <td>26.794185</td>
          <td>1.123420</td>
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
          <td>22.823508</td>
          <td>23.643726</td>
          <td>27.155182</td>
          <td>26.115826</td>
          <td>21.502638</td>
          <td>28.778459</td>
          <td>16.448378</td>
          <td>0.005002</td>
          <td>22.049986</td>
          <td>0.023037</td>
          <td>23.511810</td>
          <td>0.091950</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.624859</td>
          <td>25.264117</td>
          <td>22.450229</td>
          <td>21.450214</td>
          <td>22.630703</td>
          <td>26.118446</td>
          <td>21.705103</td>
          <td>0.020327</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.209577</td>
          <td>0.012470</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.505806</td>
          <td>25.450549</td>
          <td>27.564398</td>
          <td>22.949995</td>
          <td>20.536780</td>
          <td>22.976313</td>
          <td>21.440245</td>
          <td>0.016254</td>
          <td>22.149427</td>
          <td>0.025125</td>
          <td>21.629335</td>
          <td>0.017508</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178153</td>
          <td>25.263392</td>
          <td>24.644918</td>
          <td>19.303609</td>
          <td>21.526562</td>
          <td>25.128546</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.782012</td>
          <td>0.005733</td>
          <td>19.459788</td>
          <td>0.005498</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.340817</td>
          <td>26.562563</td>
          <td>26.636875</td>
          <td>21.263552</td>
          <td>23.461059</td>
          <td>28.890656</td>
          <td>25.081212</td>
          <td>0.373599</td>
          <td>23.565902</td>
          <td>0.088304</td>
          <td>24.068209</td>
          <td>0.149297</td>
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

