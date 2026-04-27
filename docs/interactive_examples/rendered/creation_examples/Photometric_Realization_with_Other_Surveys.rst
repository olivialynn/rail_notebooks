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
    /home/runner/.cache/lephare/runs/20260427T122115


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
      File "/tmp/ipykernel_5926/2313627096.py", line 5, in <module>
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
          <td>20.739768</td>
          <td>18.917811</td>
          <td>23.716169</td>
          <td>23.783728</td>
          <td>25.886576</td>
          <td>21.682258</td>
          <td>23.863764</td>
          <td>22.524614</td>
          <td>24.386421</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.571625</td>
          <td>21.824621</td>
          <td>25.581153</td>
          <td>22.991909</td>
          <td>25.810264</td>
          <td>19.432619</td>
          <td>21.182473</td>
          <td>23.211510</td>
          <td>27.705725</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.812391</td>
          <td>22.622859</td>
          <td>22.886708</td>
          <td>23.043669</td>
          <td>28.414104</td>
          <td>23.418726</td>
          <td>23.152529</td>
          <td>21.279887</td>
          <td>20.836678</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.315846</td>
          <td>19.516655</td>
          <td>20.779506</td>
          <td>24.384540</td>
          <td>21.260061</td>
          <td>24.569163</td>
          <td>20.588888</td>
          <td>20.254440</td>
          <td>23.433407</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.123491</td>
          <td>24.949562</td>
          <td>27.166496</td>
          <td>22.117789</td>
          <td>20.896179</td>
          <td>27.268209</td>
          <td>23.000423</td>
          <td>22.769935</td>
          <td>20.199322</td>
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
          <td>23.812695</td>
          <td>21.833641</td>
          <td>20.890205</td>
          <td>27.493999</td>
          <td>20.747598</td>
          <td>24.075983</td>
          <td>20.160660</td>
          <td>20.287594</td>
          <td>24.718085</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990476</td>
          <td>18.311398</td>
          <td>23.780779</td>
          <td>22.282952</td>
          <td>22.519102</td>
          <td>26.065268</td>
          <td>24.007713</td>
          <td>26.396980</td>
          <td>21.146566</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.770976</td>
          <td>22.890479</td>
          <td>23.231792</td>
          <td>23.491471</td>
          <td>18.751312</td>
          <td>21.886268</td>
          <td>23.425274</td>
          <td>20.615843</td>
          <td>21.133086</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.154042</td>
          <td>25.127339</td>
          <td>20.577094</td>
          <td>15.736358</td>
          <td>24.326023</td>
          <td>21.575695</td>
          <td>19.954573</td>
          <td>22.902157</td>
          <td>20.360044</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.158507</td>
          <td>26.152923</td>
          <td>26.107903</td>
          <td>23.781399</td>
          <td>20.374943</td>
          <td>27.059614</td>
          <td>16.531193</td>
          <td>22.105090</td>
          <td>19.446801</td>
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
          <td>20.736631</td>
          <td>0.005747</td>
          <td>18.911722</td>
          <td>0.005011</td>
          <td>23.705299</td>
          <td>0.011163</td>
          <td>23.783457</td>
          <td>0.018338</td>
          <td>25.886775</td>
          <td>0.216968</td>
          <td>21.701839</td>
          <td>0.012986</td>
          <td>23.863764</td>
          <td>22.524614</td>
          <td>24.386421</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.552286</td>
          <td>0.013425</td>
          <td>21.828984</td>
          <td>0.005502</td>
          <td>25.561717</td>
          <td>0.053474</td>
          <td>22.991276</td>
          <td>0.009980</td>
          <td>25.671604</td>
          <td>0.181074</td>
          <td>19.442943</td>
          <td>0.005273</td>
          <td>21.182473</td>
          <td>23.211510</td>
          <td>27.705725</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.813406</td>
          <td>0.005831</td>
          <td>22.623410</td>
          <td>0.006692</td>
          <td>22.871891</td>
          <td>0.006894</td>
          <td>23.047950</td>
          <td>0.010373</td>
          <td>26.881445</td>
          <td>0.477591</td>
          <td>23.520082</td>
          <td>0.061865</td>
          <td>23.152529</td>
          <td>21.279887</td>
          <td>20.836678</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.130981</td>
          <td>0.121302</td>
          <td>19.514661</td>
          <td>0.005021</td>
          <td>20.780163</td>
          <td>0.005070</td>
          <td>24.423566</td>
          <td>0.031859</td>
          <td>21.261393</td>
          <td>0.006150</td>
          <td>24.340378</td>
          <td>0.127197</td>
          <td>20.588888</td>
          <td>20.254440</td>
          <td>23.433407</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.902324</td>
          <td>0.099444</td>
          <td>24.947297</td>
          <td>0.035347</td>
          <td>27.134164</td>
          <td>0.209999</td>
          <td>22.119469</td>
          <td>0.006391</td>
          <td>20.892036</td>
          <td>0.005638</td>
          <td>26.032838</td>
          <td>0.503249</td>
          <td>23.000423</td>
          <td>22.769935</td>
          <td>20.199322</td>
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
          <td>23.850126</td>
          <td>0.039484</td>
          <td>21.835141</td>
          <td>0.005506</td>
          <td>20.887611</td>
          <td>0.005083</td>
          <td>28.097308</td>
          <td>0.674921</td>
          <td>20.746142</td>
          <td>0.005505</td>
          <td>24.091707</td>
          <td>0.102424</td>
          <td>20.160660</td>
          <td>20.287594</td>
          <td>24.718085</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.004200</td>
          <td>0.019165</td>
          <td>18.314441</td>
          <td>0.005006</td>
          <td>23.760895</td>
          <td>0.011623</td>
          <td>22.277162</td>
          <td>0.006776</td>
          <td>22.527879</td>
          <td>0.012100</td>
          <td>25.581477</td>
          <td>0.356887</td>
          <td>24.007713</td>
          <td>26.396980</td>
          <td>21.146566</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.124666</td>
          <td>0.279869</td>
          <td>22.887619</td>
          <td>0.007497</td>
          <td>23.239626</td>
          <td>0.008261</td>
          <td>23.516124</td>
          <td>0.014731</td>
          <td>18.754450</td>
          <td>0.005025</td>
          <td>21.891188</td>
          <td>0.015069</td>
          <td>23.425274</td>
          <td>20.615843</td>
          <td>21.133086</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.872057</td>
          <td>0.227544</td>
          <td>25.156866</td>
          <td>0.042528</td>
          <td>20.574743</td>
          <td>0.005052</td>
          <td>15.739058</td>
          <td>0.005000</td>
          <td>24.390579</td>
          <td>0.059245</td>
          <td>21.594421</td>
          <td>0.011974</td>
          <td>19.954573</td>
          <td>22.902157</td>
          <td>20.360044</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.181684</td>
          <td>0.022209</td>
          <td>26.107728</td>
          <td>0.098481</td>
          <td>26.196647</td>
          <td>0.093755</td>
          <td>23.768814</td>
          <td>0.018115</td>
          <td>20.374280</td>
          <td>0.005278</td>
          <td>25.683664</td>
          <td>0.386479</td>
          <td>16.531193</td>
          <td>22.105090</td>
          <td>19.446801</td>
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
          <td>20.739768</td>
          <td>18.917811</td>
          <td>23.716169</td>
          <td>23.783728</td>
          <td>25.886576</td>
          <td>21.682258</td>
          <td>23.853020</td>
          <td>0.012897</td>
          <td>22.518445</td>
          <td>0.007853</td>
          <td>24.303542</td>
          <td>0.031427</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.571625</td>
          <td>21.824621</td>
          <td>25.581153</td>
          <td>22.991909</td>
          <td>25.810264</td>
          <td>19.432619</td>
          <td>21.186640</td>
          <td>0.005104</td>
          <td>23.223800</td>
          <td>0.012607</td>
          <td>29.292234</td>
          <td>1.469356</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.812391</td>
          <td>22.622859</td>
          <td>22.886708</td>
          <td>23.043669</td>
          <td>28.414104</td>
          <td>23.418726</td>
          <td>23.153616</td>
          <td>0.008008</td>
          <td>21.282100</td>
          <td>0.005364</td>
          <td>20.849810</td>
          <td>0.005167</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.315846</td>
          <td>19.516655</td>
          <td>20.779506</td>
          <td>24.384540</td>
          <td>21.260061</td>
          <td>24.569163</td>
          <td>20.590894</td>
          <td>0.005035</td>
          <td>20.252688</td>
          <td>0.005056</td>
          <td>23.439673</td>
          <td>0.014965</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.123491</td>
          <td>24.949562</td>
          <td>27.166496</td>
          <td>22.117789</td>
          <td>20.896179</td>
          <td>27.268209</td>
          <td>23.001454</td>
          <td>0.007387</td>
          <td>22.787062</td>
          <td>0.009224</td>
          <td>20.196851</td>
          <td>0.005051</td>
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
          <td>23.812695</td>
          <td>21.833641</td>
          <td>20.890205</td>
          <td>27.493999</td>
          <td>20.747598</td>
          <td>24.075983</td>
          <td>20.175249</td>
          <td>0.005016</td>
          <td>20.280991</td>
          <td>0.005059</td>
          <td>24.792261</td>
          <td>0.048544</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990476</td>
          <td>18.311398</td>
          <td>23.780779</td>
          <td>22.282952</td>
          <td>22.519102</td>
          <td>26.065268</td>
          <td>24.011167</td>
          <td>0.014623</td>
          <td>26.251880</td>
          <td>0.174684</td>
          <td>21.148869</td>
          <td>0.005287</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.770976</td>
          <td>22.890479</td>
          <td>23.231792</td>
          <td>23.491471</td>
          <td>18.751312</td>
          <td>21.886268</td>
          <td>23.426306</td>
          <td>0.009464</td>
          <td>20.618041</td>
          <td>0.005110</td>
          <td>21.134138</td>
          <td>0.005279</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.154042</td>
          <td>25.127339</td>
          <td>20.577094</td>
          <td>15.736358</td>
          <td>24.326023</td>
          <td>21.575695</td>
          <td>19.952933</td>
          <td>0.005011</td>
          <td>22.891050</td>
          <td>0.009886</td>
          <td>20.357591</td>
          <td>0.005068</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.158507</td>
          <td>26.152923</td>
          <td>26.107903</td>
          <td>23.781399</td>
          <td>20.374943</td>
          <td>27.059614</td>
          <td>16.524011</td>
          <td>0.005000</td>
          <td>22.106773</td>
          <td>0.006496</td>
          <td>19.447098</td>
          <td>0.005013</td>
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
          <td>20.739768</td>
          <td>18.917811</td>
          <td>23.716169</td>
          <td>23.783728</td>
          <td>25.886576</td>
          <td>21.682258</td>
          <td>23.862110</td>
          <td>0.136246</td>
          <td>22.521681</td>
          <td>0.034899</td>
          <td>24.271816</td>
          <td>0.177668</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.571625</td>
          <td>21.824621</td>
          <td>25.581153</td>
          <td>22.991909</td>
          <td>25.810264</td>
          <td>19.432619</td>
          <td>21.183484</td>
          <td>0.013207</td>
          <td>23.196970</td>
          <td>0.063682</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.812391</td>
          <td>22.622859</td>
          <td>22.886708</td>
          <td>23.043669</td>
          <td>28.414104</td>
          <td>23.418726</td>
          <td>23.325974</td>
          <td>0.085247</td>
          <td>21.294324</td>
          <td>0.012325</td>
          <td>20.830366</td>
          <td>0.009490</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.315846</td>
          <td>19.516655</td>
          <td>20.779506</td>
          <td>24.384540</td>
          <td>21.260061</td>
          <td>24.569163</td>
          <td>20.587493</td>
          <td>0.008662</td>
          <td>20.258377</td>
          <td>0.006626</td>
          <td>23.426484</td>
          <td>0.085285</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.123491</td>
          <td>24.949562</td>
          <td>27.166496</td>
          <td>22.117789</td>
          <td>20.896179</td>
          <td>27.268209</td>
          <td>23.006073</td>
          <td>0.064199</td>
          <td>22.782806</td>
          <td>0.044031</td>
          <td>20.199846</td>
          <td>0.006739</td>
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
          <td>23.812695</td>
          <td>21.833641</td>
          <td>20.890205</td>
          <td>27.493999</td>
          <td>20.747598</td>
          <td>24.075983</td>
          <td>20.156453</td>
          <td>0.006903</td>
          <td>20.288603</td>
          <td>0.006708</td>
          <td>24.787019</td>
          <td>0.272840</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990476</td>
          <td>18.311398</td>
          <td>23.780779</td>
          <td>22.282952</td>
          <td>22.519102</td>
          <td>26.065268</td>
          <td>24.152469</td>
          <td>0.174771</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.138043</td>
          <td>0.011809</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.770976</td>
          <td>22.890479</td>
          <td>23.231792</td>
          <td>23.491471</td>
          <td>18.751312</td>
          <td>21.886268</td>
          <td>23.359948</td>
          <td>0.087841</td>
          <td>20.630563</td>
          <td>0.007906</td>
          <td>21.147766</td>
          <td>0.011896</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.154042</td>
          <td>25.127339</td>
          <td>20.577094</td>
          <td>15.736358</td>
          <td>24.326023</td>
          <td>21.575695</td>
          <td>19.956205</td>
          <td>0.006377</td>
          <td>22.980746</td>
          <td>0.052529</td>
          <td>20.360301</td>
          <td>0.007240</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.158507</td>
          <td>26.152923</td>
          <td>26.107903</td>
          <td>23.781399</td>
          <td>20.374943</td>
          <td>27.059614</td>
          <td>16.531249</td>
          <td>0.005003</td>
          <td>22.131069</td>
          <td>0.024725</td>
          <td>19.459799</td>
          <td>0.005498</td>
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

