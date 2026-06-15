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
    /home/runner/.cache/lephare/runs/20260615T135044


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
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelapp.py", line 807, in start
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
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 460, in do_execute
        res = shell.run_cell(
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/zmqshell.py", line 665, in run_cell
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
      File "/tmp/ipykernel_4066/2313627096.py", line 5, in <module>
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
          <td>22.427973</td>
          <td>26.877787</td>
          <td>27.730111</td>
          <td>28.671594</td>
          <td>18.658077</td>
          <td>21.228375</td>
          <td>22.518922</td>
          <td>21.018100</td>
          <td>29.170808</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.816057</td>
          <td>22.676412</td>
          <td>22.196276</td>
          <td>22.723573</td>
          <td>20.234282</td>
          <td>20.026804</td>
          <td>20.727871</td>
          <td>21.528060</td>
          <td>20.836111</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.095618</td>
          <td>20.174471</td>
          <td>16.809273</td>
          <td>24.135322</td>
          <td>21.306064</td>
          <td>23.077319</td>
          <td>19.778009</td>
          <td>20.244297</td>
          <td>23.528546</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.930716</td>
          <td>22.625384</td>
          <td>24.899371</td>
          <td>17.449096</td>
          <td>18.831006</td>
          <td>22.451312</td>
          <td>19.234333</td>
          <td>19.230006</td>
          <td>26.668275</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.650954</td>
          <td>23.325552</td>
          <td>28.235328</td>
          <td>20.126222</td>
          <td>24.382812</td>
          <td>19.449134</td>
          <td>25.403797</td>
          <td>23.887130</td>
          <td>24.412040</td>
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
          <td>21.689874</td>
          <td>28.372955</td>
          <td>24.501188</td>
          <td>25.248786</td>
          <td>19.978289</td>
          <td>21.646053</td>
          <td>22.678743</td>
          <td>23.862537</td>
          <td>24.369363</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.019362</td>
          <td>22.082611</td>
          <td>20.799886</td>
          <td>22.448914</td>
          <td>17.331315</td>
          <td>22.433668</td>
          <td>23.290528</td>
          <td>18.653657</td>
          <td>23.928449</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916894</td>
          <td>27.519023</td>
          <td>28.245081</td>
          <td>17.829411</td>
          <td>24.459964</td>
          <td>22.054495</td>
          <td>20.936807</td>
          <td>16.429445</td>
          <td>21.937790</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.230584</td>
          <td>26.761468</td>
          <td>22.095467</td>
          <td>18.946764</td>
          <td>20.498310</td>
          <td>25.821918</td>
          <td>24.802725</td>
          <td>20.321971</td>
          <td>25.578756</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.210327</td>
          <td>25.577702</td>
          <td>26.196246</td>
          <td>25.585144</td>
          <td>22.114244</td>
          <td>27.409021</td>
          <td>27.459850</td>
          <td>23.637433</td>
          <td>24.218268</td>
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
          <td>22.426326</td>
          <td>0.012240</td>
          <td>26.731612</td>
          <td>0.168878</td>
          <td>29.019168</td>
          <td>0.860942</td>
          <td>27.707059</td>
          <td>0.511500</td>
          <td>18.655470</td>
          <td>0.005022</td>
          <td>21.233333</td>
          <td>0.009324</td>
          <td>22.518922</td>
          <td>21.018100</td>
          <td>29.170808</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.804718</td>
          <td>0.037947</td>
          <td>22.674145</td>
          <td>0.006825</td>
          <td>22.194815</td>
          <td>0.005651</td>
          <td>22.718843</td>
          <td>0.008415</td>
          <td>20.242507</td>
          <td>0.005225</td>
          <td>20.021739</td>
          <td>0.005691</td>
          <td>20.727871</td>
          <td>21.528060</td>
          <td>20.836111</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.091956</td>
          <td>0.005312</td>
          <td>20.171422</td>
          <td>0.005047</td>
          <td>16.807751</td>
          <td>0.005001</td>
          <td>24.152787</td>
          <td>0.025139</td>
          <td>21.303142</td>
          <td>0.006228</td>
          <td>23.113095</td>
          <td>0.043111</td>
          <td>19.778009</td>
          <td>20.244297</td>
          <td>23.528546</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.917182</td>
          <td>0.017852</td>
          <td>22.612458</td>
          <td>0.006665</td>
          <td>24.919211</td>
          <td>0.030280</td>
          <td>17.457942</td>
          <td>0.005002</td>
          <td>18.826161</td>
          <td>0.005028</td>
          <td>22.426813</td>
          <td>0.023595</td>
          <td>19.234333</td>
          <td>19.230006</td>
          <td>26.668275</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.652647</td>
          <td>0.005020</td>
          <td>23.324189</td>
          <td>0.009594</td>
          <td>29.823046</td>
          <td>1.371498</td>
          <td>20.134626</td>
          <td>0.005061</td>
          <td>24.354609</td>
          <td>0.057384</td>
          <td>19.453876</td>
          <td>0.005278</td>
          <td>25.403797</td>
          <td>23.887130</td>
          <td>24.412040</td>
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
          <td>21.688772</td>
          <td>0.007802</td>
          <td>28.487707</td>
          <td>0.665966</td>
          <td>24.517992</td>
          <td>0.021375</td>
          <td>25.191812</td>
          <td>0.062938</td>
          <td>19.980264</td>
          <td>0.005149</td>
          <td>21.639641</td>
          <td>0.012386</td>
          <td>22.678743</td>
          <td>23.862537</td>
          <td>24.369363</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.014122</td>
          <td>0.045590</td>
          <td>22.082369</td>
          <td>0.005741</td>
          <td>20.799005</td>
          <td>0.005072</td>
          <td>22.451299</td>
          <td>0.007311</td>
          <td>17.332549</td>
          <td>0.005005</td>
          <td>22.424876</td>
          <td>0.023556</td>
          <td>23.290528</td>
          <td>18.653657</td>
          <td>23.928449</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916936</td>
          <td>0.005961</td>
          <td>27.403483</td>
          <td>0.294938</td>
          <td>28.770550</td>
          <td>0.731937</td>
          <td>17.822365</td>
          <td>0.005003</td>
          <td>24.461964</td>
          <td>0.063117</td>
          <td>22.049955</td>
          <td>0.017151</td>
          <td>20.936807</td>
          <td>16.429445</td>
          <td>21.937790</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>27.157475</td>
          <td>0.241337</td>
          <td>22.087618</td>
          <td>0.005548</td>
          <td>18.953289</td>
          <td>0.005012</td>
          <td>20.498736</td>
          <td>0.005339</td>
          <td>26.214190</td>
          <td>0.574073</td>
          <td>24.802725</td>
          <td>20.321971</td>
          <td>25.578756</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.305184</td>
          <td>0.058891</td>
          <td>25.548024</td>
          <td>0.060130</td>
          <td>26.200088</td>
          <td>0.094039</td>
          <td>25.591858</td>
          <td>0.089627</td>
          <td>22.105998</td>
          <td>0.009053</td>
          <td>27.047689</td>
          <td>0.995359</td>
          <td>27.459850</td>
          <td>23.637433</td>
          <td>24.218268</td>
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
          <td>22.427973</td>
          <td>26.877787</td>
          <td>27.730111</td>
          <td>28.671594</td>
          <td>18.658077</td>
          <td>21.228375</td>
          <td>22.526104</td>
          <td>0.006110</td>
          <td>21.016835</td>
          <td>0.005226</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.816057</td>
          <td>22.676412</td>
          <td>22.196276</td>
          <td>22.723573</td>
          <td>20.234282</td>
          <td>20.026804</td>
          <td>20.724555</td>
          <td>0.005044</td>
          <td>21.529489</td>
          <td>0.005563</td>
          <td>20.834062</td>
          <td>0.005162</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.095618</td>
          <td>20.174471</td>
          <td>16.809273</td>
          <td>24.135322</td>
          <td>21.306064</td>
          <td>23.077319</td>
          <td>19.778144</td>
          <td>0.005008</td>
          <td>20.237760</td>
          <td>0.005055</td>
          <td>23.469598</td>
          <td>0.015335</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.930716</td>
          <td>22.625384</td>
          <td>24.899371</td>
          <td>17.449096</td>
          <td>18.831006</td>
          <td>22.451312</td>
          <td>19.226784</td>
          <td>0.005003</td>
          <td>19.228604</td>
          <td>0.005009</td>
          <td>26.241934</td>
          <td>0.173213</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.650954</td>
          <td>23.325552</td>
          <td>28.235328</td>
          <td>20.126222</td>
          <td>24.382812</td>
          <td>19.449134</td>
          <td>25.325755</td>
          <td>0.045750</td>
          <td>23.872430</td>
          <td>0.021540</td>
          <td>24.354360</td>
          <td>0.032874</td>
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
          <td>21.689874</td>
          <td>28.372955</td>
          <td>24.501188</td>
          <td>25.248786</td>
          <td>19.978289</td>
          <td>21.646053</td>
          <td>22.679369</td>
          <td>0.006431</td>
          <td>23.837681</td>
          <td>0.020904</td>
          <td>24.357821</td>
          <td>0.032975</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.019362</td>
          <td>22.082611</td>
          <td>20.799886</td>
          <td>22.448914</td>
          <td>17.331315</td>
          <td>22.433668</td>
          <td>23.288199</td>
          <td>0.008666</td>
          <td>18.661256</td>
          <td>0.005003</td>
          <td>23.930289</td>
          <td>0.022647</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916894</td>
          <td>27.519023</td>
          <td>28.245081</td>
          <td>17.829411</td>
          <td>24.459964</td>
          <td>22.054495</td>
          <td>20.937551</td>
          <td>0.005066</td>
          <td>16.432580</td>
          <td>0.005000</td>
          <td>21.931769</td>
          <td>0.006121</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.230584</td>
          <td>26.761468</td>
          <td>22.095467</td>
          <td>18.946764</td>
          <td>20.498310</td>
          <td>25.821918</td>
          <td>24.826546</td>
          <td>0.029359</td>
          <td>20.327102</td>
          <td>0.005064</td>
          <td>25.606611</td>
          <td>0.099944</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.210327</td>
          <td>25.577702</td>
          <td>26.196246</td>
          <td>25.585144</td>
          <td>22.114244</td>
          <td>27.409021</td>
          <td>27.363655</td>
          <td>0.267693</td>
          <td>23.615171</td>
          <td>0.017301</td>
          <td>24.219196</td>
          <td>0.029169</td>
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
          <td>22.427973</td>
          <td>26.877787</td>
          <td>27.730111</td>
          <td>28.671594</td>
          <td>18.658077</td>
          <td>21.228375</td>
          <td>22.613729</td>
          <td>0.045262</td>
          <td>21.006577</td>
          <td>0.009992</td>
          <td>25.810874</td>
          <td>0.597297</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.816057</td>
          <td>22.676412</td>
          <td>22.196276</td>
          <td>22.723573</td>
          <td>20.234282</td>
          <td>20.026804</td>
          <td>20.722368</td>
          <td>0.009440</td>
          <td>21.524707</td>
          <td>0.014784</td>
          <td>20.828438</td>
          <td>0.009478</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.095618</td>
          <td>20.174471</td>
          <td>16.809273</td>
          <td>24.135322</td>
          <td>21.306064</td>
          <td>23.077319</td>
          <td>19.780353</td>
          <td>0.006028</td>
          <td>20.238727</td>
          <td>0.006576</td>
          <td>23.724386</td>
          <td>0.110805</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.930716</td>
          <td>22.625384</td>
          <td>24.899371</td>
          <td>17.449096</td>
          <td>18.831006</td>
          <td>22.451312</td>
          <td>19.224784</td>
          <td>0.005392</td>
          <td>19.240045</td>
          <td>0.005282</td>
          <td>26.165003</td>
          <td>0.761405</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.650954</td>
          <td>23.325552</td>
          <td>28.235328</td>
          <td>20.126222</td>
          <td>24.382812</td>
          <td>19.449134</td>
          <td>25.504848</td>
          <td>0.514844</td>
          <td>23.900597</td>
          <td>0.118424</td>
          <td>24.392129</td>
          <td>0.196694</td>
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
          <td>21.689874</td>
          <td>28.372955</td>
          <td>24.501188</td>
          <td>25.248786</td>
          <td>19.978289</td>
          <td>21.646053</td>
          <td>22.636631</td>
          <td>0.046195</td>
          <td>23.837551</td>
          <td>0.112087</td>
          <td>24.514613</td>
          <td>0.217958</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.019362</td>
          <td>22.082611</td>
          <td>20.799886</td>
          <td>22.448914</td>
          <td>17.331315</td>
          <td>22.433668</td>
          <td>23.166045</td>
          <td>0.073998</td>
          <td>18.656844</td>
          <td>0.005098</td>
          <td>24.114093</td>
          <td>0.155295</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916894</td>
          <td>27.519023</td>
          <td>28.245081</td>
          <td>17.829411</td>
          <td>24.459964</td>
          <td>22.054495</td>
          <td>20.940871</td>
          <td>0.010989</td>
          <td>16.439245</td>
          <td>0.005002</td>
          <td>21.898947</td>
          <td>0.022040</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.230584</td>
          <td>26.761468</td>
          <td>22.095467</td>
          <td>18.946764</td>
          <td>20.498310</td>
          <td>25.821918</td>
          <td>25.038405</td>
          <td>0.361311</td>
          <td>20.319240</td>
          <td>0.006793</td>
          <td>26.169207</td>
          <td>0.763526</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.210327</td>
          <td>25.577702</td>
          <td>26.196246</td>
          <td>25.585144</td>
          <td>22.114244</td>
          <td>27.409021</td>
          <td>26.368012</td>
          <td>0.924662</td>
          <td>23.623345</td>
          <td>0.092889</td>
          <td>23.935819</td>
          <td>0.133182</td>
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

