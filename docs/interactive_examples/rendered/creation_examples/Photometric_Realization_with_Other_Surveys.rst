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
    /home/runner/.cache/lephare/runs/20260324T150130


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
      File "/tmp/ipykernel_7654/2313627096.py", line 5, in <module>
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
          <td>19.802648</td>
          <td>18.204593</td>
          <td>24.552384</td>
          <td>26.805210</td>
          <td>15.096233</td>
          <td>19.328322</td>
          <td>22.375673</td>
          <td>23.032950</td>
          <td>27.148853</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.585585</td>
          <td>19.658290</td>
          <td>24.667835</td>
          <td>28.142161</td>
          <td>21.638175</td>
          <td>20.364721</td>
          <td>21.511250</td>
          <td>26.184999</td>
          <td>21.219888</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.413097</td>
          <td>24.963696</td>
          <td>22.192132</td>
          <td>19.999657</td>
          <td>19.877141</td>
          <td>23.300679</td>
          <td>23.897115</td>
          <td>19.928412</td>
          <td>22.534050</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.130905</td>
          <td>25.536460</td>
          <td>23.639319</td>
          <td>25.459240</td>
          <td>19.357891</td>
          <td>19.379997</td>
          <td>19.613697</td>
          <td>28.268797</td>
          <td>20.284601</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.974961</td>
          <td>21.103618</td>
          <td>15.800438</td>
          <td>21.567466</td>
          <td>21.867515</td>
          <td>20.759076</td>
          <td>21.711732</td>
          <td>22.697670</td>
          <td>22.121875</td>
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
          <td>23.498965</td>
          <td>28.395731</td>
          <td>26.207585</td>
          <td>19.431149</td>
          <td>23.693015</td>
          <td>29.228911</td>
          <td>19.191669</td>
          <td>26.526580</td>
          <td>21.428362</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.078708</td>
          <td>19.584726</td>
          <td>19.469551</td>
          <td>25.463036</td>
          <td>21.522228</td>
          <td>19.390675</td>
          <td>24.374373</td>
          <td>25.142529</td>
          <td>25.183343</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.061398</td>
          <td>24.103218</td>
          <td>24.937188</td>
          <td>23.378353</td>
          <td>21.771085</td>
          <td>20.656747</td>
          <td>23.474198</td>
          <td>21.526737</td>
          <td>21.203100</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.343054</td>
          <td>22.902669</td>
          <td>17.423672</td>
          <td>20.983446</td>
          <td>23.796553</td>
          <td>24.765976</td>
          <td>21.789675</td>
          <td>18.133353</td>
          <td>23.376032</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.437215</td>
          <td>23.621445</td>
          <td>25.451232</td>
          <td>23.083477</td>
          <td>19.249187</td>
          <td>19.954466</td>
          <td>20.682140</td>
          <td>25.449610</td>
          <td>24.894444</td>
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
          <td>19.809089</td>
          <td>0.005217</td>
          <td>18.204156</td>
          <td>0.005005</td>
          <td>24.549260</td>
          <td>0.021955</td>
          <td>26.613583</td>
          <td>0.215869</td>
          <td>15.099395</td>
          <td>0.005001</td>
          <td>19.328767</td>
          <td>0.005228</td>
          <td>22.375673</td>
          <td>23.032950</td>
          <td>27.148853</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.590592</td>
          <td>0.007452</td>
          <td>19.657836</td>
          <td>0.005025</td>
          <td>24.654477</td>
          <td>0.024037</td>
          <td>27.316798</td>
          <td>0.380830</td>
          <td>21.642040</td>
          <td>0.007069</td>
          <td>20.366867</td>
          <td>0.006197</td>
          <td>21.511250</td>
          <td>26.184999</td>
          <td>21.219888</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.434177</td>
          <td>0.065969</td>
          <td>24.942931</td>
          <td>0.035211</td>
          <td>22.193409</td>
          <td>0.005650</td>
          <td>19.997592</td>
          <td>0.005050</td>
          <td>19.876844</td>
          <td>0.005127</td>
          <td>23.308705</td>
          <td>0.051284</td>
          <td>23.897115</td>
          <td>19.928412</td>
          <td>22.534050</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.127849</td>
          <td>0.006291</td>
          <td>25.477173</td>
          <td>0.056474</td>
          <td>23.626401</td>
          <td>0.010557</td>
          <td>25.363876</td>
          <td>0.073298</td>
          <td>19.357478</td>
          <td>0.005059</td>
          <td>19.382472</td>
          <td>0.005248</td>
          <td>19.613697</td>
          <td>28.268797</td>
          <td>20.284601</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.972330</td>
          <td>0.005267</td>
          <td>21.102528</td>
          <td>0.005168</td>
          <td>15.812785</td>
          <td>0.005000</td>
          <td>21.570838</td>
          <td>0.005580</td>
          <td>21.862944</td>
          <td>0.007870</td>
          <td>20.761716</td>
          <td>0.007195</td>
          <td>21.711732</td>
          <td>22.697670</td>
          <td>22.121875</td>
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
          <td>23.465100</td>
          <td>0.028262</td>
          <td>27.264120</td>
          <td>0.263411</td>
          <td>26.450052</td>
          <td>0.117011</td>
          <td>19.433864</td>
          <td>0.005023</td>
          <td>23.674764</td>
          <td>0.031427</td>
          <td>28.131020</td>
          <td>1.762781</td>
          <td>19.191669</td>
          <td>26.526580</td>
          <td>21.428362</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.056045</td>
          <td>0.020001</td>
          <td>19.582539</td>
          <td>0.005023</td>
          <td>19.467622</td>
          <td>0.005012</td>
          <td>25.508243</td>
          <td>0.083265</td>
          <td>21.508371</td>
          <td>0.006689</td>
          <td>19.400447</td>
          <td>0.005255</td>
          <td>24.374373</td>
          <td>25.142529</td>
          <td>25.183343</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.868699</td>
          <td>0.978603</td>
          <td>24.109657</td>
          <td>0.017226</td>
          <td>24.895197</td>
          <td>0.029648</td>
          <td>23.390237</td>
          <td>0.013341</td>
          <td>21.777847</td>
          <td>0.007534</td>
          <td>20.656529</td>
          <td>0.006873</td>
          <td>23.474198</td>
          <td>21.526737</td>
          <td>21.203100</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.339832</td>
          <td>0.006735</td>
          <td>22.904680</td>
          <td>0.007559</td>
          <td>17.419511</td>
          <td>0.005001</td>
          <td>20.972471</td>
          <td>0.005221</td>
          <td>23.812407</td>
          <td>0.035481</td>
          <td>24.830562</td>
          <td>0.193440</td>
          <td>21.789675</td>
          <td>18.133353</td>
          <td>23.376032</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.442164</td>
          <td>0.005006</td>
          <td>23.611218</td>
          <td>0.011684</td>
          <td>25.446653</td>
          <td>0.048280</td>
          <td>23.093353</td>
          <td>0.010706</td>
          <td>19.246697</td>
          <td>0.005050</td>
          <td>19.958744</td>
          <td>0.005625</td>
          <td>20.682140</td>
          <td>25.449610</td>
          <td>24.894444</td>
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
          <td>19.802648</td>
          <td>18.204593</td>
          <td>24.552384</td>
          <td>26.805210</td>
          <td>15.096233</td>
          <td>19.328322</td>
          <td>22.387291</td>
          <td>0.005878</td>
          <td>23.042115</td>
          <td>0.010999</td>
          <td>27.577302</td>
          <td>0.504525</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.585585</td>
          <td>19.658290</td>
          <td>24.667835</td>
          <td>28.142161</td>
          <td>21.638175</td>
          <td>20.364721</td>
          <td>21.508207</td>
          <td>0.005186</td>
          <td>26.219744</td>
          <td>0.169971</td>
          <td>21.223762</td>
          <td>0.005328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.413097</td>
          <td>24.963696</td>
          <td>22.192132</td>
          <td>19.999657</td>
          <td>19.877141</td>
          <td>23.300679</td>
          <td>23.902814</td>
          <td>0.013410</td>
          <td>19.920173</td>
          <td>0.005031</td>
          <td>22.529266</td>
          <td>0.007900</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.130905</td>
          <td>25.536460</td>
          <td>23.639319</td>
          <td>25.459240</td>
          <td>19.357891</td>
          <td>19.379997</td>
          <td>19.611312</td>
          <td>0.005006</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.284311</td>
          <td>0.005060</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.974961</td>
          <td>21.103618</td>
          <td>15.800438</td>
          <td>21.567466</td>
          <td>21.867515</td>
          <td>20.759076</td>
          <td>21.706240</td>
          <td>0.005265</td>
          <td>22.706265</td>
          <td>0.008763</td>
          <td>22.127634</td>
          <td>0.006548</td>
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
          <td>23.498965</td>
          <td>28.395731</td>
          <td>26.207585</td>
          <td>19.431149</td>
          <td>23.693015</td>
          <td>29.228911</td>
          <td>19.188808</td>
          <td>0.005003</td>
          <td>26.588324</td>
          <td>0.231736</td>
          <td>21.433484</td>
          <td>0.005475</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.078708</td>
          <td>19.584726</td>
          <td>19.469551</td>
          <td>25.463036</td>
          <td>21.522228</td>
          <td>19.390675</td>
          <td>24.383568</td>
          <td>0.019956</td>
          <td>24.993897</td>
          <td>0.058100</td>
          <td>25.113550</td>
          <td>0.064628</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.061398</td>
          <td>24.103218</td>
          <td>24.937188</td>
          <td>23.378353</td>
          <td>21.771085</td>
          <td>20.656747</td>
          <td>23.458038</td>
          <td>0.009667</td>
          <td>21.531331</td>
          <td>0.005564</td>
          <td>21.209912</td>
          <td>0.005320</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.343054</td>
          <td>22.902669</td>
          <td>17.423672</td>
          <td>20.983446</td>
          <td>23.796553</td>
          <td>24.765976</td>
          <td>21.790696</td>
          <td>0.005309</td>
          <td>18.139070</td>
          <td>0.005001</td>
          <td>23.376799</td>
          <td>0.014224</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.437215</td>
          <td>23.621445</td>
          <td>25.451232</td>
          <td>23.083477</td>
          <td>19.249187</td>
          <td>19.954466</td>
          <td>20.674684</td>
          <td>0.005041</td>
          <td>25.313371</td>
          <td>0.077167</td>
          <td>24.917386</td>
          <td>0.054272</td>
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
          <td>19.802648</td>
          <td>18.204593</td>
          <td>24.552384</td>
          <td>26.805210</td>
          <td>15.096233</td>
          <td>19.328322</td>
          <td>22.361458</td>
          <td>0.036154</td>
          <td>23.073344</td>
          <td>0.057046</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.585585</td>
          <td>19.658290</td>
          <td>24.667835</td>
          <td>28.142161</td>
          <td>21.638175</td>
          <td>20.364721</td>
          <td>21.505893</td>
          <td>0.017167</td>
          <td>26.595411</td>
          <td>0.940454</td>
          <td>21.226203</td>
          <td>0.012631</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.413097</td>
          <td>24.963696</td>
          <td>22.192132</td>
          <td>19.999657</td>
          <td>19.877141</td>
          <td>23.300679</td>
          <td>23.891477</td>
          <td>0.139747</td>
          <td>19.929313</td>
          <td>0.005943</td>
          <td>22.587704</td>
          <td>0.040453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.130905</td>
          <td>25.536460</td>
          <td>23.639319</td>
          <td>25.459240</td>
          <td>19.357891</td>
          <td>19.379997</td>
          <td>19.623042</td>
          <td>0.005787</td>
          <td>28.995443</td>
          <td>2.831153</td>
          <td>20.283729</td>
          <td>0.006987</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.974961</td>
          <td>21.103618</td>
          <td>15.800438</td>
          <td>21.567466</td>
          <td>21.867515</td>
          <td>20.759076</td>
          <td>21.727666</td>
          <td>0.020725</td>
          <td>22.706746</td>
          <td>0.041145</td>
          <td>22.096059</td>
          <td>0.026173</td>
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
          <td>23.498965</td>
          <td>28.395731</td>
          <td>26.207585</td>
          <td>19.431149</td>
          <td>23.693015</td>
          <td>29.228911</td>
          <td>19.197366</td>
          <td>0.005374</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.419567</td>
          <td>0.014723</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.078708</td>
          <td>19.584726</td>
          <td>19.469551</td>
          <td>25.463036</td>
          <td>21.522228</td>
          <td>19.390675</td>
          <td>24.231957</td>
          <td>0.186957</td>
          <td>25.263015</td>
          <td>0.368333</td>
          <td>25.074760</td>
          <td>0.343673</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.061398</td>
          <td>24.103218</td>
          <td>24.937188</td>
          <td>23.378353</td>
          <td>21.771085</td>
          <td>20.656747</td>
          <td>23.470504</td>
          <td>0.096823</td>
          <td>21.484970</td>
          <td>0.014317</td>
          <td>21.201365</td>
          <td>0.012391</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.343054</td>
          <td>22.902669</td>
          <td>17.423672</td>
          <td>20.983446</td>
          <td>23.796553</td>
          <td>24.765976</td>
          <td>21.768778</td>
          <td>0.021472</td>
          <td>18.136210</td>
          <td>0.005038</td>
          <td>23.440993</td>
          <td>0.086384</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.437215</td>
          <td>23.621445</td>
          <td>25.451232</td>
          <td>23.083477</td>
          <td>19.249187</td>
          <td>19.954466</td>
          <td>20.670310</td>
          <td>0.009125</td>
          <td>26.142150</td>
          <td>0.701220</td>
          <td>24.630662</td>
          <td>0.240000</td>
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

