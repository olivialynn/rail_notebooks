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
    /home/runner/.cache/lephare/runs/20260511T124646


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
      File "/tmp/ipykernel_5592/2313627096.py", line 5, in <module>
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
          <td>18.102056</td>
          <td>23.765346</td>
          <td>22.921563</td>
          <td>23.197949</td>
          <td>20.753834</td>
          <td>26.448486</td>
          <td>20.531082</td>
          <td>20.598557</td>
          <td>26.267586</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.380919</td>
          <td>25.908091</td>
          <td>26.034175</td>
          <td>25.602183</td>
          <td>24.801539</td>
          <td>21.597397</td>
          <td>18.971462</td>
          <td>25.005478</td>
          <td>23.291641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.207373</td>
          <td>25.742130</td>
          <td>16.705751</td>
          <td>22.382219</td>
          <td>21.186818</td>
          <td>21.043168</td>
          <td>22.734170</td>
          <td>24.812266</td>
          <td>21.649095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.045160</td>
          <td>22.941827</td>
          <td>27.130281</td>
          <td>23.201525</td>
          <td>19.570298</td>
          <td>21.573621</td>
          <td>24.290475</td>
          <td>24.606352</td>
          <td>18.115492</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.242054</td>
          <td>28.245370</td>
          <td>24.258756</td>
          <td>24.433554</td>
          <td>21.042043</td>
          <td>21.618768</td>
          <td>24.025153</td>
          <td>20.924247</td>
          <td>21.516188</td>
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
          <td>20.749950</td>
          <td>20.319545</td>
          <td>25.837644</td>
          <td>23.681884</td>
          <td>22.420576</td>
          <td>23.667069</td>
          <td>22.133038</td>
          <td>23.781412</td>
          <td>26.441266</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.850629</td>
          <td>20.696413</td>
          <td>21.517918</td>
          <td>25.789878</td>
          <td>20.310847</td>
          <td>15.979960</td>
          <td>23.534145</td>
          <td>18.931521</td>
          <td>25.629937</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.984454</td>
          <td>25.291106</td>
          <td>18.029631</td>
          <td>23.967656</td>
          <td>23.207107</td>
          <td>22.097443</td>
          <td>25.839218</td>
          <td>19.102203</td>
          <td>23.405180</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.827534</td>
          <td>21.644357</td>
          <td>24.281480</td>
          <td>19.408711</td>
          <td>21.163501</td>
          <td>21.915748</td>
          <td>21.923144</td>
          <td>26.476913</td>
          <td>26.616438</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.927270</td>
          <td>19.432836</td>
          <td>23.891881</td>
          <td>21.517334</td>
          <td>18.432704</td>
          <td>24.868024</td>
          <td>23.916743</td>
          <td>21.732456</td>
          <td>17.731487</td>
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
          <td>18.106648</td>
          <td>0.005031</td>
          <td>23.761970</td>
          <td>0.013072</td>
          <td>22.928526</td>
          <td>0.007063</td>
          <td>23.196234</td>
          <td>0.011526</td>
          <td>20.759987</td>
          <td>0.005516</td>
          <td>26.868515</td>
          <td>0.891419</td>
          <td>20.531082</td>
          <td>20.598557</td>
          <td>26.267586</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.367826</td>
          <td>0.062228</td>
          <td>25.813349</td>
          <td>0.076025</td>
          <td>26.039925</td>
          <td>0.081674</td>
          <td>25.517739</td>
          <td>0.083964</td>
          <td>24.823533</td>
          <td>0.086883</td>
          <td>21.608720</td>
          <td>0.012102</td>
          <td>18.971462</td>
          <td>25.005478</td>
          <td>23.291641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.251778</td>
          <td>0.056187</td>
          <td>25.754829</td>
          <td>0.072197</td>
          <td>16.701755</td>
          <td>0.005001</td>
          <td>22.379858</td>
          <td>0.007076</td>
          <td>21.184407</td>
          <td>0.006018</td>
          <td>21.042429</td>
          <td>0.008310</td>
          <td>22.734170</td>
          <td>24.812266</td>
          <td>21.649095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.043913</td>
          <td>0.005029</td>
          <td>22.934781</td>
          <td>0.007673</td>
          <td>27.091821</td>
          <td>0.202681</td>
          <td>23.188050</td>
          <td>0.011457</td>
          <td>19.564291</td>
          <td>0.005079</td>
          <td>21.569865</td>
          <td>0.011759</td>
          <td>24.290475</td>
          <td>24.606352</td>
          <td>18.115492</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.228544</td>
          <td>0.010670</td>
          <td>28.241225</td>
          <td>0.559891</td>
          <td>24.268656</td>
          <td>0.017326</td>
          <td>24.447033</td>
          <td>0.032524</td>
          <td>21.031649</td>
          <td>0.005798</td>
          <td>21.604237</td>
          <td>0.012062</td>
          <td>24.025153</td>
          <td>20.924247</td>
          <td>21.516188</td>
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
          <td>20.740491</td>
          <td>0.005751</td>
          <td>20.318802</td>
          <td>0.005057</td>
          <td>25.968630</td>
          <td>0.076692</td>
          <td>23.682123</td>
          <td>0.016857</td>
          <td>22.419149</td>
          <td>0.011173</td>
          <td>23.594430</td>
          <td>0.066079</td>
          <td>22.133038</td>
          <td>23.781412</td>
          <td>26.441266</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.857996</td>
          <td>0.005231</td>
          <td>20.702023</td>
          <td>0.005095</td>
          <td>21.516008</td>
          <td>0.005219</td>
          <td>25.743426</td>
          <td>0.102378</td>
          <td>20.306743</td>
          <td>0.005250</td>
          <td>15.980713</td>
          <td>0.005003</td>
          <td>23.534145</td>
          <td>18.931521</td>
          <td>25.629937</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.984200</td>
          <td>0.006056</td>
          <td>25.302258</td>
          <td>0.048368</td>
          <td>18.031190</td>
          <td>0.005002</td>
          <td>23.982980</td>
          <td>0.021714</td>
          <td>23.192105</td>
          <td>0.020677</td>
          <td>22.126619</td>
          <td>0.018279</td>
          <td>25.839218</td>
          <td>19.102203</td>
          <td>23.405180</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.843351</td>
          <td>0.016823</td>
          <td>21.644734</td>
          <td>0.005378</td>
          <td>24.265362</td>
          <td>0.017279</td>
          <td>19.403580</td>
          <td>0.005022</td>
          <td>21.162101</td>
          <td>0.005983</td>
          <td>21.900087</td>
          <td>0.015177</td>
          <td>21.923144</td>
          <td>26.476913</td>
          <td>26.616438</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.928003</td>
          <td>0.005976</td>
          <td>19.426905</td>
          <td>0.005019</td>
          <td>23.897813</td>
          <td>0.012880</td>
          <td>21.518327</td>
          <td>0.005533</td>
          <td>18.430628</td>
          <td>0.005017</td>
          <td>25.039515</td>
          <td>0.230350</td>
          <td>23.916743</td>
          <td>21.732456</td>
          <td>17.731487</td>
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
          <td>18.102056</td>
          <td>23.765346</td>
          <td>22.921563</td>
          <td>23.197949</td>
          <td>20.753834</td>
          <td>26.448486</td>
          <td>20.532055</td>
          <td>0.005031</td>
          <td>20.594562</td>
          <td>0.005105</td>
          <td>25.887455</td>
          <td>0.127715</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.380919</td>
          <td>25.908091</td>
          <td>26.034175</td>
          <td>25.602183</td>
          <td>24.801539</td>
          <td>21.597397</td>
          <td>18.961621</td>
          <td>0.005002</td>
          <td>25.142562</td>
          <td>0.066316</td>
          <td>23.293006</td>
          <td>0.013307</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.207373</td>
          <td>25.742130</td>
          <td>16.705751</td>
          <td>22.382219</td>
          <td>21.186818</td>
          <td>21.043168</td>
          <td>22.733326</td>
          <td>0.006562</td>
          <td>24.838190</td>
          <td>0.050573</td>
          <td>21.645453</td>
          <td>0.005688</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.045160</td>
          <td>22.941827</td>
          <td>27.130281</td>
          <td>23.201525</td>
          <td>19.570298</td>
          <td>21.573621</td>
          <td>24.298592</td>
          <td>0.018562</td>
          <td>24.574009</td>
          <td>0.039963</td>
          <td>18.112578</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.242054</td>
          <td>28.245370</td>
          <td>24.258756</td>
          <td>24.433554</td>
          <td>21.042043</td>
          <td>21.618768</td>
          <td>24.019464</td>
          <td>0.014722</td>
          <td>20.925593</td>
          <td>0.005192</td>
          <td>21.514414</td>
          <td>0.005548</td>
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
          <td>20.749950</td>
          <td>20.319545</td>
          <td>25.837644</td>
          <td>23.681884</td>
          <td>22.420576</td>
          <td>23.667069</td>
          <td>22.131812</td>
          <td>0.005565</td>
          <td>23.778061</td>
          <td>0.019862</td>
          <td>26.802307</td>
          <td>0.276255</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.850629</td>
          <td>20.696413</td>
          <td>21.517918</td>
          <td>25.789878</td>
          <td>20.310847</td>
          <td>15.979960</td>
          <td>23.544740</td>
          <td>0.010260</td>
          <td>18.936539</td>
          <td>0.005005</td>
          <td>25.610174</td>
          <td>0.100257</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.984454</td>
          <td>25.291106</td>
          <td>18.029631</td>
          <td>23.967656</td>
          <td>23.207107</td>
          <td>22.097443</td>
          <td>25.851148</td>
          <td>0.073027</td>
          <td>19.110878</td>
          <td>0.005007</td>
          <td>23.432069</td>
          <td>0.014873</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.827534</td>
          <td>21.644357</td>
          <td>24.281480</td>
          <td>19.408711</td>
          <td>21.163501</td>
          <td>21.915748</td>
          <td>21.926356</td>
          <td>0.005393</td>
          <td>26.544372</td>
          <td>0.223429</td>
          <td>26.692243</td>
          <td>0.252489</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.927270</td>
          <td>19.432836</td>
          <td>23.891881</td>
          <td>21.517334</td>
          <td>18.432704</td>
          <td>24.868024</td>
          <td>23.927888</td>
          <td>0.013679</td>
          <td>21.739292</td>
          <td>0.005809</td>
          <td>17.734325</td>
          <td>0.005001</td>
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
          <td>18.102056</td>
          <td>23.765346</td>
          <td>22.921563</td>
          <td>23.197949</td>
          <td>20.753834</td>
          <td>26.448486</td>
          <td>20.533915</td>
          <td>0.008387</td>
          <td>20.605439</td>
          <td>0.007798</td>
          <td>25.837259</td>
          <td>0.608539</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.380919</td>
          <td>25.908091</td>
          <td>26.034175</td>
          <td>25.602183</td>
          <td>24.801539</td>
          <td>21.597397</td>
          <td>18.969642</td>
          <td>0.005249</td>
          <td>25.129867</td>
          <td>0.331676</td>
          <td>23.333840</td>
          <td>0.078578</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.207373</td>
          <td>25.742130</td>
          <td>16.705751</td>
          <td>22.382219</td>
          <td>21.186818</td>
          <td>21.043168</td>
          <td>22.784495</td>
          <td>0.052704</td>
          <td>25.368768</td>
          <td>0.399824</td>
          <td>21.626842</td>
          <td>0.017471</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.045160</td>
          <td>22.941827</td>
          <td>27.130281</td>
          <td>23.201525</td>
          <td>19.570298</td>
          <td>21.573621</td>
          <td>24.672824</td>
          <td>0.269703</td>
          <td>24.791025</td>
          <td>0.252237</td>
          <td>18.129513</td>
          <td>0.005045</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.242054</td>
          <td>28.245370</td>
          <td>24.258756</td>
          <td>24.433554</td>
          <td>21.042043</td>
          <td>21.618768</td>
          <td>23.807830</td>
          <td>0.129991</td>
          <td>20.907450</td>
          <td>0.009348</td>
          <td>21.499194</td>
          <td>0.015712</td>
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
          <td>20.749950</td>
          <td>20.319545</td>
          <td>25.837644</td>
          <td>23.681884</td>
          <td>22.420576</td>
          <td>23.667069</td>
          <td>22.137742</td>
          <td>0.029650</td>
          <td>23.811023</td>
          <td>0.109518</td>
          <td>26.686840</td>
          <td>1.055445</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.850629</td>
          <td>20.696413</td>
          <td>21.517918</td>
          <td>25.789878</td>
          <td>20.310847</td>
          <td>15.979960</td>
          <td>23.564406</td>
          <td>0.105140</td>
          <td>18.933947</td>
          <td>0.005162</td>
          <td>24.678808</td>
          <td>0.249716</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.984454</td>
          <td>25.291106</td>
          <td>18.029631</td>
          <td>23.967656</td>
          <td>23.207107</td>
          <td>22.097443</td>
          <td>26.998619</td>
          <td>1.329169</td>
          <td>19.100215</td>
          <td>0.005219</td>
          <td>23.494057</td>
          <td>0.090523</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.827534</td>
          <td>21.644357</td>
          <td>24.281480</td>
          <td>19.408711</td>
          <td>21.163501</td>
          <td>21.915748</td>
          <td>21.940722</td>
          <td>0.024934</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.927270</td>
          <td>19.432836</td>
          <td>23.891881</td>
          <td>21.517334</td>
          <td>18.432704</td>
          <td>24.868024</td>
          <td>23.589950</td>
          <td>0.107518</td>
          <td>21.737563</td>
          <td>0.017629</td>
          <td>17.731381</td>
          <td>0.005022</td>
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

