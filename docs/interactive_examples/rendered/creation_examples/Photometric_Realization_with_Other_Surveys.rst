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
    /home/runner/.cache/lephare/runs/20260330T121200


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
      File "/tmp/ipykernel_4507/2313627096.py", line 5, in <module>
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
          <td>25.158410</td>
          <td>26.428820</td>
          <td>21.235544</td>
          <td>26.475781</td>
          <td>25.384395</td>
          <td>23.426047</td>
          <td>20.444273</td>
          <td>27.828994</td>
          <td>24.533160</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725877</td>
          <td>23.364264</td>
          <td>22.574552</td>
          <td>20.419931</td>
          <td>26.079592</td>
          <td>24.172376</td>
          <td>18.693603</td>
          <td>25.005377</td>
          <td>28.285372</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.871449</td>
          <td>27.509040</td>
          <td>17.221252</td>
          <td>21.806970</td>
          <td>25.640960</td>
          <td>17.557927</td>
          <td>22.514384</td>
          <td>22.090543</td>
          <td>26.745973</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.085992</td>
          <td>21.603664</td>
          <td>21.050621</td>
          <td>26.363906</td>
          <td>26.807448</td>
          <td>26.449990</td>
          <td>21.285647</td>
          <td>26.962786</td>
          <td>14.725978</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.409328</td>
          <td>24.069411</td>
          <td>24.032976</td>
          <td>26.611527</td>
          <td>20.759609</td>
          <td>22.783919</td>
          <td>24.253809</td>
          <td>25.180020</td>
          <td>21.708348</td>
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
          <td>23.499394</td>
          <td>27.233449</td>
          <td>24.759093</td>
          <td>22.481929</td>
          <td>24.392453</td>
          <td>17.338863</td>
          <td>22.839166</td>
          <td>23.979818</td>
          <td>23.873521</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.261575</td>
          <td>20.515566</td>
          <td>26.051652</td>
          <td>19.359791</td>
          <td>22.371358</td>
          <td>24.286261</td>
          <td>21.415500</td>
          <td>21.973337</td>
          <td>27.365467</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.126717</td>
          <td>25.810512</td>
          <td>21.974106</td>
          <td>21.040688</td>
          <td>23.416731</td>
          <td>19.356548</td>
          <td>26.136301</td>
          <td>22.468420</td>
          <td>21.793506</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.948085</td>
          <td>24.557908</td>
          <td>20.783117</td>
          <td>19.484679</td>
          <td>21.721664</td>
          <td>23.776478</td>
          <td>22.640802</td>
          <td>22.117342</td>
          <td>18.629229</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.797214</td>
          <td>19.429430</td>
          <td>23.748771</td>
          <td>22.116794</td>
          <td>21.679032</td>
          <td>23.151914</td>
          <td>21.388391</td>
          <td>25.209391</td>
          <td>22.853005</td>
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
          <td>25.131597</td>
          <td>0.121367</td>
          <td>26.196666</td>
          <td>0.106444</td>
          <td>21.236299</td>
          <td>0.005141</td>
          <td>26.107344</td>
          <td>0.140460</td>
          <td>25.651639</td>
          <td>0.178037</td>
          <td>23.499427</td>
          <td>0.060742</td>
          <td>20.444273</td>
          <td>27.828994</td>
          <td>24.533160</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.718499</td>
          <td>0.015243</td>
          <td>23.368526</td>
          <td>0.009875</td>
          <td>22.565096</td>
          <td>0.006177</td>
          <td>20.420048</td>
          <td>0.005093</td>
          <td>26.231204</td>
          <td>0.287924</td>
          <td>24.156518</td>
          <td>0.108395</td>
          <td>18.693603</td>
          <td>25.005377</td>
          <td>28.285372</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.794375</td>
          <td>0.037606</td>
          <td>27.008219</td>
          <td>0.213227</td>
          <td>17.217130</td>
          <td>0.005001</td>
          <td>21.806978</td>
          <td>0.005848</td>
          <td>25.742757</td>
          <td>0.192293</td>
          <td>17.558247</td>
          <td>0.005018</td>
          <td>22.514384</td>
          <td>22.090543</td>
          <td>26.745973</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.082088</td>
          <td>0.009708</td>
          <td>21.600817</td>
          <td>0.005354</td>
          <td>21.044707</td>
          <td>0.005105</td>
          <td>26.311833</td>
          <td>0.167365</td>
          <td>26.949132</td>
          <td>0.502156</td>
          <td>32.716235</td>
          <td>6.113374</td>
          <td>21.285647</td>
          <td>26.962786</td>
          <td>14.725978</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.605140</td>
          <td>0.182004</td>
          <td>24.066441</td>
          <td>0.016628</td>
          <td>24.030583</td>
          <td>0.014283</td>
          <td>26.671738</td>
          <td>0.226574</td>
          <td>20.758286</td>
          <td>0.005515</td>
          <td>22.807248</td>
          <td>0.032893</td>
          <td>24.253809</td>
          <td>25.180020</td>
          <td>21.708348</td>
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
          <td>23.501395</td>
          <td>0.029159</td>
          <td>27.051655</td>
          <td>0.221086</td>
          <td>24.821988</td>
          <td>0.027807</td>
          <td>22.481923</td>
          <td>0.007419</td>
          <td>24.396559</td>
          <td>0.059560</td>
          <td>17.336717</td>
          <td>0.005014</td>
          <td>22.839166</td>
          <td>23.979818</td>
          <td>23.873521</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.303102</td>
          <td>0.024607</td>
          <td>20.513703</td>
          <td>0.005074</td>
          <td>26.015629</td>
          <td>0.079942</td>
          <td>19.364160</td>
          <td>0.005021</td>
          <td>22.364148</td>
          <td>0.010745</td>
          <td>24.269853</td>
          <td>0.119645</td>
          <td>21.415500</td>
          <td>21.973337</td>
          <td>27.365467</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.324412</td>
          <td>2.047075</td>
          <td>25.812752</td>
          <td>0.075985</td>
          <td>21.968066</td>
          <td>0.005452</td>
          <td>21.042114</td>
          <td>0.005247</td>
          <td>23.402679</td>
          <td>0.024778</td>
          <td>19.359579</td>
          <td>0.005239</td>
          <td>26.136301</td>
          <td>22.468420</td>
          <td>21.793506</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.692635</td>
          <td>0.195918</td>
          <td>24.573363</td>
          <td>0.025497</td>
          <td>20.785258</td>
          <td>0.005071</td>
          <td>19.483857</td>
          <td>0.005024</td>
          <td>21.738964</td>
          <td>0.007392</td>
          <td>23.738779</td>
          <td>0.075082</td>
          <td>22.640802</td>
          <td>22.117342</td>
          <td>18.629229</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.667863</td>
          <td>0.863892</td>
          <td>19.434573</td>
          <td>0.005019</td>
          <td>23.738529</td>
          <td>0.011435</td>
          <td>22.115222</td>
          <td>0.006382</td>
          <td>21.673876</td>
          <td>0.007171</td>
          <td>23.173852</td>
          <td>0.045498</td>
          <td>21.388391</td>
          <td>25.209391</td>
          <td>22.853005</td>
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
          <td>25.158410</td>
          <td>26.428820</td>
          <td>21.235544</td>
          <td>26.475781</td>
          <td>25.384395</td>
          <td>23.426047</td>
          <td>20.441376</td>
          <td>0.005026</td>
          <td>28.353920</td>
          <td>0.860728</td>
          <td>24.544421</td>
          <td>0.038924</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725877</td>
          <td>23.364264</td>
          <td>22.574552</td>
          <td>20.419931</td>
          <td>26.079592</td>
          <td>24.172376</td>
          <td>18.702888</td>
          <td>0.005001</td>
          <td>25.087937</td>
          <td>0.063172</td>
          <td>26.908349</td>
          <td>0.300990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.871449</td>
          <td>27.509040</td>
          <td>17.221252</td>
          <td>21.806970</td>
          <td>25.640960</td>
          <td>17.557927</td>
          <td>22.505363</td>
          <td>0.006072</td>
          <td>22.096713</td>
          <td>0.006472</td>
          <td>26.746546</td>
          <td>0.263979</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.085992</td>
          <td>21.603664</td>
          <td>21.050621</td>
          <td>26.363906</td>
          <td>26.807448</td>
          <td>26.449990</td>
          <td>21.284955</td>
          <td>0.005124</td>
          <td>26.709411</td>
          <td>0.256073</td>
          <td>14.724122</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.409328</td>
          <td>24.069411</td>
          <td>24.032976</td>
          <td>26.611527</td>
          <td>20.759609</td>
          <td>22.783919</td>
          <td>24.247612</td>
          <td>0.017779</td>
          <td>25.137475</td>
          <td>0.066017</td>
          <td>21.699167</td>
          <td>0.005755</td>
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
          <td>23.499394</td>
          <td>27.233449</td>
          <td>24.759093</td>
          <td>22.481929</td>
          <td>24.392453</td>
          <td>17.338863</td>
          <td>22.842938</td>
          <td>0.006862</td>
          <td>23.995644</td>
          <td>0.023972</td>
          <td>23.884655</td>
          <td>0.021769</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.261575</td>
          <td>20.515566</td>
          <td>26.051652</td>
          <td>19.359791</td>
          <td>22.371358</td>
          <td>24.286261</td>
          <td>21.411658</td>
          <td>0.005156</td>
          <td>21.976096</td>
          <td>0.006207</td>
          <td>26.946619</td>
          <td>0.310378</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.126717</td>
          <td>25.810512</td>
          <td>21.974106</td>
          <td>21.040688</td>
          <td>23.416731</td>
          <td>19.356548</td>
          <td>26.210655</td>
          <td>0.100299</td>
          <td>22.463134</td>
          <td>0.007624</td>
          <td>21.793989</td>
          <td>0.005888</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.948085</td>
          <td>24.557908</td>
          <td>20.783117</td>
          <td>19.484679</td>
          <td>21.721664</td>
          <td>23.776478</td>
          <td>22.638293</td>
          <td>0.006337</td>
          <td>22.118691</td>
          <td>0.006525</td>
          <td>18.621840</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.797214</td>
          <td>19.429430</td>
          <td>23.748771</td>
          <td>22.116794</td>
          <td>21.679032</td>
          <td>23.151914</td>
          <td>21.393795</td>
          <td>0.005151</td>
          <td>25.217246</td>
          <td>0.070863</td>
          <td>22.849132</td>
          <td>0.009609</td>
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
          <td>25.158410</td>
          <td>26.428820</td>
          <td>21.235544</td>
          <td>26.475781</td>
          <td>25.384395</td>
          <td>23.426047</td>
          <td>20.440711</td>
          <td>0.007950</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.704318</td>
          <td>0.255005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725877</td>
          <td>23.364264</td>
          <td>22.574552</td>
          <td>20.419931</td>
          <td>26.079592</td>
          <td>24.172376</td>
          <td>18.694441</td>
          <td>0.005151</td>
          <td>24.747835</td>
          <td>0.243426</td>
          <td>27.512233</td>
          <td>1.636641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.871449</td>
          <td>27.509040</td>
          <td>17.221252</td>
          <td>21.806970</td>
          <td>25.640960</td>
          <td>17.557927</td>
          <td>22.455231</td>
          <td>0.039300</td>
          <td>22.074751</td>
          <td>0.023539</td>
          <td>26.232499</td>
          <td>0.795950</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.085992</td>
          <td>21.603664</td>
          <td>21.050621</td>
          <td>26.363906</td>
          <td>26.807448</td>
          <td>26.449990</td>
          <td>21.264011</td>
          <td>0.014078</td>
          <td>25.451733</td>
          <td>0.426061</td>
          <td>14.727951</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.409328</td>
          <td>24.069411</td>
          <td>24.032976</td>
          <td>26.611527</td>
          <td>20.759609</td>
          <td>22.783919</td>
          <td>24.393356</td>
          <td>0.214124</td>
          <td>24.936714</td>
          <td>0.284078</td>
          <td>21.691312</td>
          <td>0.018448</td>
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
          <td>23.499394</td>
          <td>27.233449</td>
          <td>24.759093</td>
          <td>22.481929</td>
          <td>24.392453</td>
          <td>17.338863</td>
          <td>22.759260</td>
          <td>0.051532</td>
          <td>24.009855</td>
          <td>0.130220</td>
          <td>23.911195</td>
          <td>0.130371</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.261575</td>
          <td>20.515566</td>
          <td>26.051652</td>
          <td>19.359791</td>
          <td>22.371358</td>
          <td>24.286261</td>
          <td>21.402113</td>
          <td>0.015750</td>
          <td>21.952620</td>
          <td>0.021175</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.126717</td>
          <td>25.810512</td>
          <td>21.974106</td>
          <td>21.040688</td>
          <td>23.416731</td>
          <td>19.356548</td>
          <td>27.166294</td>
          <td>1.450178</td>
          <td>22.461939</td>
          <td>0.033096</td>
          <td>21.791113</td>
          <td>0.020085</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.948085</td>
          <td>24.557908</td>
          <td>20.783117</td>
          <td>19.484679</td>
          <td>21.721664</td>
          <td>23.776478</td>
          <td>22.645669</td>
          <td>0.046569</td>
          <td>22.102336</td>
          <td>0.024112</td>
          <td>18.623326</td>
          <td>0.005111</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.797214</td>
          <td>19.429430</td>
          <td>23.748771</td>
          <td>22.116794</td>
          <td>21.679032</td>
          <td>23.151914</td>
          <td>21.409032</td>
          <td>0.015840</td>
          <td>24.744817</td>
          <td>0.242821</td>
          <td>22.827248</td>
          <td>0.050082</td>
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

