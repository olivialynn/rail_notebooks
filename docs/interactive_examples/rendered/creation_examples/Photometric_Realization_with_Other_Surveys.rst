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
    /home/runner/.cache/lephare/runs/20260525T125859


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
      File "/tmp/ipykernel_5621/2313627096.py", line 5, in <module>
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
          <td>21.285316</td>
          <td>21.094721</td>
          <td>21.432790</td>
          <td>22.566154</td>
          <td>22.452803</td>
          <td>26.418531</td>
          <td>23.749966</td>
          <td>26.643178</td>
          <td>23.235966</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.475057</td>
          <td>23.390838</td>
          <td>22.245987</td>
          <td>19.197353</td>
          <td>23.529225</td>
          <td>20.730372</td>
          <td>29.481348</td>
          <td>21.074328</td>
          <td>28.833661</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.119887</td>
          <td>25.577168</td>
          <td>19.237260</td>
          <td>19.378879</td>
          <td>17.615420</td>
          <td>24.138641</td>
          <td>31.767448</td>
          <td>20.019464</td>
          <td>25.346275</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.753691</td>
          <td>21.413256</td>
          <td>19.106773</td>
          <td>27.474174</td>
          <td>23.369720</td>
          <td>17.626719</td>
          <td>29.609026</td>
          <td>24.521942</td>
          <td>24.515556</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.027430</td>
          <td>23.487054</td>
          <td>14.988247</td>
          <td>25.362945</td>
          <td>25.818504</td>
          <td>26.866817</td>
          <td>23.660801</td>
          <td>21.923665</td>
          <td>21.642804</td>
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
          <td>23.183117</td>
          <td>21.183128</td>
          <td>19.972698</td>
          <td>24.795818</td>
          <td>23.214169</td>
          <td>25.315726</td>
          <td>20.429886</td>
          <td>20.389288</td>
          <td>25.430859</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.810410</td>
          <td>18.752750</td>
          <td>21.320628</td>
          <td>21.385289</td>
          <td>21.095912</td>
          <td>28.235687</td>
          <td>24.784907</td>
          <td>21.332208</td>
          <td>17.510216</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.346114</td>
          <td>19.961156</td>
          <td>20.153415</td>
          <td>25.138273</td>
          <td>24.740796</td>
          <td>20.831145</td>
          <td>28.448837</td>
          <td>26.215979</td>
          <td>22.821155</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.278366</td>
          <td>21.666966</td>
          <td>21.245775</td>
          <td>23.857320</td>
          <td>22.126751</td>
          <td>24.509081</td>
          <td>27.047900</td>
          <td>26.241383</td>
          <td>19.883486</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.174827</td>
          <td>24.862049</td>
          <td>23.864860</td>
          <td>26.436635</td>
          <td>23.666077</td>
          <td>19.091257</td>
          <td>22.891580</td>
          <td>26.462231</td>
          <td>25.293333</td>
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
          <td>21.280417</td>
          <td>0.006597</td>
          <td>21.100076</td>
          <td>0.005168</td>
          <td>21.436338</td>
          <td>0.005193</td>
          <td>22.577194</td>
          <td>0.007784</td>
          <td>22.438482</td>
          <td>0.011330</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.749966</td>
          <td>26.643178</td>
          <td>23.235966</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.427149</td>
          <td>0.065563</td>
          <td>23.383745</td>
          <td>0.009974</td>
          <td>22.244206</td>
          <td>0.005705</td>
          <td>19.204871</td>
          <td>0.005017</td>
          <td>23.512546</td>
          <td>0.027263</td>
          <td>20.734827</td>
          <td>0.007109</td>
          <td>29.481348</td>
          <td>21.074328</td>
          <td>28.833661</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.113690</td>
          <td>0.006265</td>
          <td>25.664236</td>
          <td>0.066643</td>
          <td>19.232798</td>
          <td>0.005009</td>
          <td>19.386740</td>
          <td>0.005022</td>
          <td>17.616444</td>
          <td>0.005006</td>
          <td>24.170004</td>
          <td>0.109678</td>
          <td>31.767448</td>
          <td>20.019464</td>
          <td>25.346275</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.981674</td>
          <td>1.047084</td>
          <td>21.410512</td>
          <td>0.005265</td>
          <td>19.103700</td>
          <td>0.005008</td>
          <td>27.597462</td>
          <td>0.471627</td>
          <td>23.374757</td>
          <td>0.024186</td>
          <td>17.623691</td>
          <td>0.005019</td>
          <td>29.609026</td>
          <td>24.521942</td>
          <td>24.515556</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.467872</td>
          <td>0.162004</td>
          <td>23.480849</td>
          <td>0.010652</td>
          <td>14.993683</td>
          <td>0.005000</td>
          <td>25.306949</td>
          <td>0.069698</td>
          <td>26.145108</td>
          <td>0.268488</td>
          <td>27.367618</td>
          <td>1.198404</td>
          <td>23.660801</td>
          <td>21.923665</td>
          <td>21.642804</td>
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
          <td>23.206278</td>
          <td>0.022673</td>
          <td>21.182619</td>
          <td>0.005189</td>
          <td>19.971026</td>
          <td>0.005023</td>
          <td>24.707725</td>
          <td>0.040956</td>
          <td>23.242457</td>
          <td>0.021584</td>
          <td>25.155301</td>
          <td>0.253433</td>
          <td>20.429886</td>
          <td>20.389288</td>
          <td>25.430859</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.806140</td>
          <td>0.016331</td>
          <td>18.751022</td>
          <td>0.005009</td>
          <td>21.326991</td>
          <td>0.005162</td>
          <td>21.390196</td>
          <td>0.005433</td>
          <td>21.090878</td>
          <td>0.005878</td>
          <td>26.744758</td>
          <td>0.823869</td>
          <td>24.784907</td>
          <td>21.332208</td>
          <td>17.510216</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.355348</td>
          <td>0.005443</td>
          <td>19.964727</td>
          <td>0.005036</td>
          <td>20.157270</td>
          <td>0.005029</td>
          <td>25.059500</td>
          <td>0.055966</td>
          <td>24.738757</td>
          <td>0.080628</td>
          <td>20.827373</td>
          <td>0.007421</td>
          <td>28.448837</td>
          <td>26.215979</td>
          <td>22.821155</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.072065</td>
          <td>0.577841</td>
          <td>21.672290</td>
          <td>0.005395</td>
          <td>21.235525</td>
          <td>0.005141</td>
          <td>23.851933</td>
          <td>0.019424</td>
          <td>22.124334</td>
          <td>0.009157</td>
          <td>24.534492</td>
          <td>0.150378</td>
          <td>27.047900</td>
          <td>26.241383</td>
          <td>19.883486</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.166642</td>
          <td>0.021930</td>
          <td>24.832783</td>
          <td>0.031963</td>
          <td>23.866141</td>
          <td>0.012573</td>
          <td>26.209121</td>
          <td>0.153300</td>
          <td>23.591241</td>
          <td>0.029205</td>
          <td>19.086666</td>
          <td>0.005156</td>
          <td>22.891580</td>
          <td>26.462231</td>
          <td>25.293333</td>
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
          <td>21.285316</td>
          <td>21.094721</td>
          <td>21.432790</td>
          <td>22.566154</td>
          <td>22.452803</td>
          <td>26.418531</td>
          <td>23.764316</td>
          <td>0.012046</td>
          <td>26.702323</td>
          <td>0.254588</td>
          <td>23.241556</td>
          <td>0.012782</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.475057</td>
          <td>23.390838</td>
          <td>22.245987</td>
          <td>19.197353</td>
          <td>23.529225</td>
          <td>20.730372</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.070889</td>
          <td>0.005249</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.119887</td>
          <td>25.577168</td>
          <td>19.237260</td>
          <td>19.378879</td>
          <td>17.615420</td>
          <td>24.138641</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.017163</td>
          <td>0.005037</td>
          <td>25.489939</td>
          <td>0.090195</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.753691</td>
          <td>21.413256</td>
          <td>19.106773</td>
          <td>27.474174</td>
          <td>23.369720</td>
          <td>17.626719</td>
          <td>31.406921</td>
          <td>2.749408</td>
          <td>24.557488</td>
          <td>0.039379</td>
          <td>24.500588</td>
          <td>0.037435</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.027430</td>
          <td>23.487054</td>
          <td>14.988247</td>
          <td>25.362945</td>
          <td>25.818504</td>
          <td>26.866817</td>
          <td>23.668441</td>
          <td>0.011213</td>
          <td>21.926055</td>
          <td>0.006110</td>
          <td>21.637522</td>
          <td>0.005679</td>
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
          <td>23.183117</td>
          <td>21.183128</td>
          <td>19.972698</td>
          <td>24.795818</td>
          <td>23.214169</td>
          <td>25.315726</td>
          <td>20.424979</td>
          <td>0.005026</td>
          <td>20.383173</td>
          <td>0.005071</td>
          <td>25.670043</td>
          <td>0.105660</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.810410</td>
          <td>18.752750</td>
          <td>21.320628</td>
          <td>21.385289</td>
          <td>21.095912</td>
          <td>28.235687</td>
          <td>24.819386</td>
          <td>0.029174</td>
          <td>21.338268</td>
          <td>0.005402</td>
          <td>17.506538</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.346114</td>
          <td>19.961156</td>
          <td>20.153415</td>
          <td>25.138273</td>
          <td>24.740796</td>
          <td>20.831145</td>
          <td>27.768995</td>
          <td>0.370057</td>
          <td>26.100277</td>
          <td>0.153466</td>
          <td>22.817056</td>
          <td>0.009407</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.278366</td>
          <td>21.666966</td>
          <td>21.245775</td>
          <td>23.857320</td>
          <td>22.126751</td>
          <td>24.509081</td>
          <td>27.021410</td>
          <td>0.201598</td>
          <td>26.173747</td>
          <td>0.163430</td>
          <td>19.878477</td>
          <td>0.005028</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.174827</td>
          <td>24.862049</td>
          <td>23.864860</td>
          <td>26.436635</td>
          <td>23.666077</td>
          <td>19.091257</td>
          <td>22.896084</td>
          <td>0.007026</td>
          <td>26.797679</td>
          <td>0.275217</td>
          <td>25.318217</td>
          <td>0.077499</td>
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
          <td>21.285316</td>
          <td>21.094721</td>
          <td>21.432790</td>
          <td>22.566154</td>
          <td>22.452803</td>
          <td>26.418531</td>
          <td>23.794144</td>
          <td>0.128458</td>
          <td>26.761272</td>
          <td>1.039621</td>
          <td>23.319981</td>
          <td>0.077620</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.475057</td>
          <td>23.390838</td>
          <td>22.245987</td>
          <td>19.197353</td>
          <td>23.529225</td>
          <td>20.730372</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.066420</td>
          <td>0.010418</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.119887</td>
          <td>25.577168</td>
          <td>19.237260</td>
          <td>19.378879</td>
          <td>17.615420</td>
          <td>24.138641</td>
          <td>25.598109</td>
          <td>0.551007</td>
          <td>20.015516</td>
          <td>0.006091</td>
          <td>25.064420</td>
          <td>0.340878</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.753691</td>
          <td>21.413256</td>
          <td>19.106773</td>
          <td>27.474174</td>
          <td>23.369720</td>
          <td>17.626719</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.744542</td>
          <td>0.242766</td>
          <td>24.432791</td>
          <td>0.203534</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.027430</td>
          <td>23.487054</td>
          <td>14.988247</td>
          <td>25.362945</td>
          <td>25.818504</td>
          <td>26.866817</td>
          <td>23.486082</td>
          <td>0.098158</td>
          <td>21.935688</td>
          <td>0.020869</td>
          <td>21.634231</td>
          <td>0.017580</td>
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
          <td>23.183117</td>
          <td>21.183128</td>
          <td>19.972698</td>
          <td>24.795818</td>
          <td>23.214169</td>
          <td>25.315726</td>
          <td>20.453945</td>
          <td>0.008009</td>
          <td>20.388726</td>
          <td>0.007003</td>
          <td>25.259933</td>
          <td>0.397110</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.810410</td>
          <td>18.752750</td>
          <td>21.320628</td>
          <td>21.385289</td>
          <td>21.095912</td>
          <td>28.235687</td>
          <td>24.804767</td>
          <td>0.300124</td>
          <td>21.328767</td>
          <td>0.012656</td>
          <td>17.514321</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.346114</td>
          <td>19.961156</td>
          <td>20.153415</td>
          <td>25.138273</td>
          <td>24.740796</td>
          <td>20.831145</td>
          <td>26.114839</td>
          <td>0.786810</td>
          <td>27.133045</td>
          <td>1.283288</td>
          <td>22.841848</td>
          <td>0.050738</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.278366</td>
          <td>21.666966</td>
          <td>21.245775</td>
          <td>23.857320</td>
          <td>22.126751</td>
          <td>24.509081</td>
          <td>27.390352</td>
          <td>1.619645</td>
          <td>26.241188</td>
          <td>0.749467</td>
          <td>19.890959</td>
          <td>0.006046</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.174827</td>
          <td>24.862049</td>
          <td>23.864860</td>
          <td>26.436635</td>
          <td>23.666077</td>
          <td>19.091257</td>
          <td>22.867351</td>
          <td>0.056743</td>
          <td>26.174811</td>
          <td>0.716881</td>
          <td>24.995024</td>
          <td>0.322614</td>
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

