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
```02_Photometric_Realization_with_Other_Surveys.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/creation_examples/02_Photometric_Realization_with_Other_Surveys.ipynb>`__
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
    /home/runner/.cache/lephare/runs/20260323T203833


.. parsed-literal::

    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.4.3 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "<frozen runpy>", line 198, in _run_module_as_main
      File "<frozen runpy>", line 88, in _run_code
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel_launcher.py", line 18, in <module>
        app.launch_new_instance()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/traitlets/config/application.py", line 1075, in launch_instance
        app.start()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/kernelapp.py", line 758, in start
        self.io_loop.start()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/tornado/platform/asyncio.py", line 211, in start
        self.asyncio_loop.run_forever()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
        self._run_once()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
        handle._run()
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/asyncio/events.py", line 84, in _run
        self._context.run(self._callback, *self._args)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/kernelbase.py", line 621, in shell_main
        await self.dispatch_shell(msg, subshell_id=subshell_id)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/kernelbase.py", line 478, in dispatch_shell
        await result
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/ipkernel.py", line 372, in execute_request
        await super().execute_request(stream, ident, parent)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/kernelbase.py", line 834, in execute_request
        reply_content = await reply_content
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/ipkernel.py", line 464, in do_execute
        res = shell.run_cell(
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/ipykernel/zmqshell.py", line 663, in run_cell
        return super().run_cell(*args, **kwargs)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3123, in run_cell
        result = self._run_cell(
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3178, in _run_cell
        result = runner(coro)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/async_helpers.py", line 128, in _pseudo_sync_runner
        coro.send(None)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3400, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3641, in run_ast_nodes
        if await self.run_code(code, result, async_=asy):
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3701, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "/tmp/ipykernel_5428/2313627096.py", line 5, in <module>
        from rail.interactive.creation.degraders import photometric_errors
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/interactive/__init__.py", line 3, in <module>
        from . import calib, creation, estimation, evaluation, tools
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/interactive/calib/__init__.py", line 3, in <module>
        from rail.utils.interactive.initialize_utils import _initialize_interactive_module
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/utils/interactive/initialize_utils.py", line 17, in <module>
        from rail.utils.interactive.base_utils import (
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/utils/interactive/base_utils.py", line 10, in <module>
        rail.stages.import_and_attach_all(silent=True)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/stages/__init__.py", line 74, in import_and_attach_all
        RailEnv.import_all_packages(silent=silent)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/core/introspection.py", line 541, in import_all_packages
        _imported_module = importlib.import_module(pkg)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/som/__init__.py", line 1, in <module>
        from rail.creation.degraders.specz_som import *
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/rail/creation/degraders/specz_som.py", line 15, in <module>
        from somoclu import Somoclu
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/somoclu/__init__.py", line 11, in <module>
        from .train import Somoclu
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/somoclu/train.py", line 25, in <module>
        from .somoclu_wrap import train as wrap_train
      File "/opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/somoclu/somoclu_wrap.py", line 11, in <module>
        import _somoclu_wrap


::


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    File /opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/numpy/core/_multiarray_umath.py:46, in __getattr__(attr_name)
         41     # Also print the message (with traceback).  This is because old versions
         42     # of NumPy unfortunately set up the import to replace (and hide) the
         43     # error.  The traceback shouldn't be needed, but e.g. pytest plugins
         44     # seem to swallow it and we should be failing anyway...
         45     sys.stderr.write(msg + tb_msg)
    ---> 46     raise ImportError(msg)
         48 ret = getattr(_multiarray_umath, attr_name, None)
         49 if ret is None:


    ImportError: 
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.4.3 as it may crash. To support both 1.x and 2.x
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
          <td>18.320677</td>
          <td>22.351813</td>
          <td>18.020899</td>
          <td>24.327699</td>
          <td>23.959063</td>
          <td>25.840270</td>
          <td>21.657865</td>
          <td>25.091154</td>
          <td>26.265418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.937542</td>
          <td>20.560958</td>
          <td>19.190631</td>
          <td>26.045732</td>
          <td>26.655364</td>
          <td>21.480878</td>
          <td>25.727952</td>
          <td>24.486356</td>
          <td>25.311813</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.519233</td>
          <td>24.369910</td>
          <td>20.818267</td>
          <td>26.270717</td>
          <td>24.370919</td>
          <td>22.334233</td>
          <td>18.124804</td>
          <td>24.815572</td>
          <td>21.812881</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.030633</td>
          <td>24.124567</td>
          <td>25.853963</td>
          <td>15.543549</td>
          <td>16.968032</td>
          <td>25.440798</td>
          <td>19.776363</td>
          <td>19.378464</td>
          <td>23.007118</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.436440</td>
          <td>24.260215</td>
          <td>21.913165</td>
          <td>21.531512</td>
          <td>20.034205</td>
          <td>23.596033</td>
          <td>22.339704</td>
          <td>16.855469</td>
          <td>24.846130</td>
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
          <td>21.849821</td>
          <td>21.115816</td>
          <td>24.935250</td>
          <td>19.801342</td>
          <td>24.727717</td>
          <td>21.939631</td>
          <td>21.152031</td>
          <td>21.055175</td>
          <td>21.853939</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.121768</td>
          <td>22.165857</td>
          <td>23.893011</td>
          <td>21.784859</td>
          <td>19.388359</td>
          <td>20.857120</td>
          <td>20.571301</td>
          <td>22.042942</td>
          <td>26.090521</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.034752</td>
          <td>21.813675</td>
          <td>21.170410</td>
          <td>21.776342</td>
          <td>23.943249</td>
          <td>24.716026</td>
          <td>20.296134</td>
          <td>24.799177</td>
          <td>21.837255</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.467399</td>
          <td>26.428534</td>
          <td>24.711322</td>
          <td>23.247719</td>
          <td>25.206026</td>
          <td>20.464519</td>
          <td>18.405654</td>
          <td>24.289189</td>
          <td>25.229528</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.196592</td>
          <td>20.805413</td>
          <td>21.247262</td>
          <td>27.400856</td>
          <td>25.323455</td>
          <td>20.985114</td>
          <td>20.221226</td>
          <td>16.420951</td>
          <td>20.543457</td>
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
          <td>18.330402</td>
          <td>0.005040</td>
          <td>22.358639</td>
          <td>0.006133</td>
          <td>18.022271</td>
          <td>0.005002</td>
          <td>24.332480</td>
          <td>0.029408</td>
          <td>23.875131</td>
          <td>0.037504</td>
          <td>25.657927</td>
          <td>0.378839</td>
          <td>21.657865</td>
          <td>25.091154</td>
          <td>26.265418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.962755</td>
          <td>0.018526</td>
          <td>20.559772</td>
          <td>0.005079</td>
          <td>19.195182</td>
          <td>0.005009</td>
          <td>26.117159</td>
          <td>0.141653</td>
          <td>27.070019</td>
          <td>0.548485</td>
          <td>21.471564</td>
          <td>0.010951</td>
          <td>25.727952</td>
          <td>24.486356</td>
          <td>25.311813</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.514640</td>
          <td>0.005150</td>
          <td>24.357677</td>
          <td>0.021194</td>
          <td>20.817235</td>
          <td>0.005074</td>
          <td>26.154285</td>
          <td>0.146251</td>
          <td>24.416282</td>
          <td>0.060612</td>
          <td>22.301981</td>
          <td>0.021198</td>
          <td>18.124804</td>
          <td>24.815572</td>
          <td>21.812881</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.099329</td>
          <td>0.049134</td>
          <td>24.181859</td>
          <td>0.018285</td>
          <td>25.811848</td>
          <td>0.066759</td>
          <td>15.547749</td>
          <td>0.005000</td>
          <td>16.970436</td>
          <td>0.005003</td>
          <td>25.078206</td>
          <td>0.237845</td>
          <td>19.776363</td>
          <td>19.378464</td>
          <td>23.007118</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.455386</td>
          <td>0.067211</td>
          <td>24.255793</td>
          <td>0.019448</td>
          <td>21.916109</td>
          <td>0.005415</td>
          <td>21.525332</td>
          <td>0.005539</td>
          <td>20.025303</td>
          <td>0.005160</td>
          <td>23.604653</td>
          <td>0.066680</td>
          <td>22.339704</td>
          <td>16.855469</td>
          <td>24.846130</td>
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
          <td>21.849199</td>
          <td>0.008474</td>
          <td>21.110918</td>
          <td>0.005170</td>
          <td>24.919693</td>
          <td>0.030292</td>
          <td>19.794023</td>
          <td>0.005037</td>
          <td>24.699879</td>
          <td>0.077909</td>
          <td>21.975361</td>
          <td>0.016132</td>
          <td>21.152031</td>
          <td>21.055175</td>
          <td>21.853939</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.109289</td>
          <td>0.020905</td>
          <td>22.158933</td>
          <td>0.005834</td>
          <td>23.912633</td>
          <td>0.013027</td>
          <td>21.777208</td>
          <td>0.005809</td>
          <td>19.384905</td>
          <td>0.005061</td>
          <td>20.854179</td>
          <td>0.007519</td>
          <td>20.571301</td>
          <td>22.042942</td>
          <td>26.090521</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.007274</td>
          <td>0.019214</td>
          <td>21.810241</td>
          <td>0.005487</td>
          <td>21.173012</td>
          <td>0.005128</td>
          <td>21.777812</td>
          <td>0.005809</td>
          <td>23.887650</td>
          <td>0.037922</td>
          <td>24.913911</td>
          <td>0.207465</td>
          <td>20.296134</td>
          <td>24.799177</td>
          <td>21.837255</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.646459</td>
          <td>0.079496</td>
          <td>26.444467</td>
          <td>0.132011</td>
          <td>24.676705</td>
          <td>0.024504</td>
          <td>23.256327</td>
          <td>0.012049</td>
          <td>25.325654</td>
          <td>0.134674</td>
          <td>20.466058</td>
          <td>0.006397</td>
          <td>18.405654</td>
          <td>24.289189</td>
          <td>25.229528</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.226134</td>
          <td>0.054933</td>
          <td>20.810370</td>
          <td>0.005111</td>
          <td>21.246671</td>
          <td>0.005143</td>
          <td>27.746695</td>
          <td>0.526556</td>
          <td>25.680763</td>
          <td>0.182484</td>
          <td>20.970645</td>
          <td>0.007986</td>
          <td>20.221226</td>
          <td>16.420951</td>
          <td>20.543457</td>
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
          <td>18.320677</td>
          <td>22.351813</td>
          <td>18.020899</td>
          <td>24.327699</td>
          <td>23.959063</td>
          <td>25.840270</td>
          <td>21.659578</td>
          <td>0.005244</td>
          <td>25.003098</td>
          <td>0.058578</td>
          <td>26.321146</td>
          <td>0.185255</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.937542</td>
          <td>20.560958</td>
          <td>19.190631</td>
          <td>26.045732</td>
          <td>26.655364</td>
          <td>21.480878</td>
          <td>25.715651</td>
          <td>0.064748</td>
          <td>24.444213</td>
          <td>0.035604</td>
          <td>25.323593</td>
          <td>0.077869</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.519233</td>
          <td>24.369910</td>
          <td>20.818267</td>
          <td>26.270717</td>
          <td>24.370919</td>
          <td>22.334233</td>
          <td>18.128458</td>
          <td>0.005000</td>
          <td>24.768572</td>
          <td>0.047530</td>
          <td>21.809233</td>
          <td>0.005911</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.030633</td>
          <td>24.124567</td>
          <td>25.853963</td>
          <td>15.543549</td>
          <td>16.968032</td>
          <td>25.440798</td>
          <td>19.770549</td>
          <td>0.005008</td>
          <td>19.379426</td>
          <td>0.005011</td>
          <td>22.988803</td>
          <td>0.010584</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.436440</td>
          <td>24.260215</td>
          <td>21.913165</td>
          <td>21.531512</td>
          <td>20.034205</td>
          <td>23.596033</td>
          <td>22.341709</td>
          <td>0.005812</td>
          <td>16.855545</td>
          <td>0.005000</td>
          <td>24.810883</td>
          <td>0.049357</td>
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
          <td>21.849821</td>
          <td>21.115816</td>
          <td>24.935250</td>
          <td>19.801342</td>
          <td>24.727717</td>
          <td>21.939631</td>
          <td>21.147902</td>
          <td>0.005097</td>
          <td>21.048373</td>
          <td>0.005239</td>
          <td>21.845821</td>
          <td>0.005970</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.121768</td>
          <td>22.165857</td>
          <td>23.893011</td>
          <td>21.784859</td>
          <td>19.388359</td>
          <td>20.857120</td>
          <td>20.581239</td>
          <td>0.005034</td>
          <td>22.031367</td>
          <td>0.006322</td>
          <td>26.204629</td>
          <td>0.167796</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.034752</td>
          <td>21.813675</td>
          <td>21.170410</td>
          <td>21.776342</td>
          <td>23.943249</td>
          <td>24.716026</td>
          <td>20.296810</td>
          <td>0.005020</td>
          <td>24.793089</td>
          <td>0.048580</td>
          <td>21.843415</td>
          <td>0.005966</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.467399</td>
          <td>26.428534</td>
          <td>24.711322</td>
          <td>23.247719</td>
          <td>25.206026</td>
          <td>20.464519</td>
          <td>18.406815</td>
          <td>0.005001</td>
          <td>24.342130</td>
          <td>0.032520</td>
          <td>25.117204</td>
          <td>0.064838</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.196592</td>
          <td>20.805413</td>
          <td>21.247262</td>
          <td>27.400856</td>
          <td>25.323455</td>
          <td>20.985114</td>
          <td>20.214674</td>
          <td>0.005017</td>
          <td>16.426729</td>
          <td>0.005000</td>
          <td>20.544313</td>
          <td>0.005096</td>
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
          <td>18.320677</td>
          <td>22.351813</td>
          <td>18.020899</td>
          <td>24.327699</td>
          <td>23.959063</td>
          <td>25.840270</td>
          <td>21.697455</td>
          <td>0.020194</td>
          <td>26.079220</td>
          <td>0.671734</td>
          <td>26.615655</td>
          <td>1.011740</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.937542</td>
          <td>20.560958</td>
          <td>19.190631</td>
          <td>26.045732</td>
          <td>26.655364</td>
          <td>21.480878</td>
          <td>25.597348</td>
          <td>0.550704</td>
          <td>24.400881</td>
          <td>0.182103</td>
          <td>26.854397</td>
          <td>1.162617</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.519233</td>
          <td>24.369910</td>
          <td>20.818267</td>
          <td>26.270717</td>
          <td>24.370919</td>
          <td>22.334233</td>
          <td>18.122921</td>
          <td>0.005053</td>
          <td>25.212330</td>
          <td>0.353995</td>
          <td>21.805256</td>
          <td>0.020330</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.030633</td>
          <td>24.124567</td>
          <td>25.853963</td>
          <td>15.543549</td>
          <td>16.968032</td>
          <td>25.440798</td>
          <td>19.777845</td>
          <td>0.006024</td>
          <td>19.382476</td>
          <td>0.005364</td>
          <td>22.980212</td>
          <td>0.057396</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.436440</td>
          <td>24.260215</td>
          <td>21.913165</td>
          <td>21.531512</td>
          <td>20.034205</td>
          <td>23.596033</td>
          <td>22.336991</td>
          <td>0.035377</td>
          <td>16.859706</td>
          <td>0.005004</td>
          <td>24.830955</td>
          <td>0.282755</td>
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
          <td>21.849821</td>
          <td>21.115816</td>
          <td>24.935250</td>
          <td>19.801342</td>
          <td>24.727717</td>
          <td>21.939631</td>
          <td>21.164105</td>
          <td>0.013009</td>
          <td>21.065298</td>
          <td>0.010409</td>
          <td>21.863721</td>
          <td>0.021379</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.121768</td>
          <td>22.165857</td>
          <td>23.893011</td>
          <td>21.784859</td>
          <td>19.388359</td>
          <td>20.857120</td>
          <td>20.574677</td>
          <td>0.008595</td>
          <td>22.020889</td>
          <td>0.022463</td>
          <td>26.537454</td>
          <td>0.965013</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.034752</td>
          <td>21.813675</td>
          <td>21.170410</td>
          <td>21.776342</td>
          <td>23.943249</td>
          <td>24.716026</td>
          <td>20.299054</td>
          <td>0.007379</td>
          <td>25.829475</td>
          <td>0.563598</td>
          <td>21.821889</td>
          <td>0.020622</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.467399</td>
          <td>26.428534</td>
          <td>24.711322</td>
          <td>23.247719</td>
          <td>25.206026</td>
          <td>20.464519</td>
          <td>18.405336</td>
          <td>0.005089</td>
          <td>24.212013</td>
          <td>0.155018</td>
          <td>25.321128</td>
          <td>0.416223</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.196592</td>
          <td>20.805413</td>
          <td>21.247262</td>
          <td>27.400856</td>
          <td>25.323455</td>
          <td>20.985114</td>
          <td>20.218187</td>
          <td>0.007097</td>
          <td>16.432207</td>
          <td>0.005002</td>
          <td>20.531173</td>
          <td>0.007908</td>
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

