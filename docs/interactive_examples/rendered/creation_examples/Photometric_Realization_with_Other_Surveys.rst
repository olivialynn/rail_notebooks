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
    /home/runner/.cache/lephare/runs/20260326T201727


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
      File "/tmp/ipykernel_4504/2313627096.py", line 5, in <module>
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
          <td>18.939778</td>
          <td>19.072641</td>
          <td>26.163908</td>
          <td>20.553186</td>
          <td>21.799546</td>
          <td>21.114998</td>
          <td>24.888560</td>
          <td>22.729915</td>
          <td>27.937383</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.388917</td>
          <td>20.036402</td>
          <td>20.117551</td>
          <td>26.215399</td>
          <td>21.132268</td>
          <td>22.241236</td>
          <td>24.704619</td>
          <td>27.117632</td>
          <td>24.112935</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.276329</td>
          <td>18.860751</td>
          <td>25.106899</td>
          <td>28.275819</td>
          <td>20.681661</td>
          <td>18.706773</td>
          <td>25.705488</td>
          <td>19.631577</td>
          <td>24.023759</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.978888</td>
          <td>23.160220</td>
          <td>30.082883</td>
          <td>20.308177</td>
          <td>22.481479</td>
          <td>21.212916</td>
          <td>19.398884</td>
          <td>24.627587</td>
          <td>23.089795</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.325230</td>
          <td>23.412856</td>
          <td>20.921770</td>
          <td>19.978369</td>
          <td>27.239717</td>
          <td>24.703266</td>
          <td>25.057373</td>
          <td>27.254548</td>
          <td>25.291235</td>
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
          <td>15.145201</td>
          <td>22.872954</td>
          <td>18.393378</td>
          <td>20.489487</td>
          <td>20.622421</td>
          <td>20.489121</td>
          <td>25.385269</td>
          <td>28.606679</td>
          <td>24.419141</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.884553</td>
          <td>25.767245</td>
          <td>23.337443</td>
          <td>21.738545</td>
          <td>14.398790</td>
          <td>26.188011</td>
          <td>27.491173</td>
          <td>21.812410</td>
          <td>18.880427</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.158100</td>
          <td>20.846878</td>
          <td>24.948462</td>
          <td>22.433533</td>
          <td>25.192708</td>
          <td>21.361917</td>
          <td>24.091214</td>
          <td>23.163089</td>
          <td>23.181289</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.579910</td>
          <td>25.722199</td>
          <td>23.289484</td>
          <td>22.704111</td>
          <td>22.113844</td>
          <td>14.438278</td>
          <td>25.341520</td>
          <td>23.577715</td>
          <td>28.622641</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.602941</td>
          <td>26.179746</td>
          <td>20.217659</td>
          <td>26.744500</td>
          <td>21.119197</td>
          <td>24.263217</td>
          <td>24.706970</td>
          <td>21.233421</td>
          <td>20.396110</td>
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
          <td>18.939268</td>
          <td>0.005077</td>
          <td>19.079653</td>
          <td>0.005013</td>
          <td>26.321858</td>
          <td>0.104629</td>
          <td>20.554610</td>
          <td>0.005115</td>
          <td>21.805436</td>
          <td>0.007639</td>
          <td>21.111188</td>
          <td>0.008648</td>
          <td>24.888560</td>
          <td>22.729915</td>
          <td>27.937383</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.829887</td>
          <td>0.955727</td>
          <td>20.042018</td>
          <td>0.005040</td>
          <td>20.111188</td>
          <td>0.005027</td>
          <td>26.274428</td>
          <td>0.162109</td>
          <td>21.142360</td>
          <td>0.005952</td>
          <td>22.217153</td>
          <td>0.019724</td>
          <td>24.704619</td>
          <td>27.117632</td>
          <td>24.112935</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.291028</td>
          <td>0.011131</td>
          <td>18.860652</td>
          <td>0.005010</td>
          <td>25.104822</td>
          <td>0.035660</td>
          <td>27.667168</td>
          <td>0.496690</td>
          <td>20.686642</td>
          <td>0.005459</td>
          <td>18.702835</td>
          <td>0.005087</td>
          <td>25.705488</td>
          <td>19.631577</td>
          <td>24.023759</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.984583</td>
          <td>0.005271</td>
          <td>23.141752</td>
          <td>0.008581</td>
          <td>28.658484</td>
          <td>0.678434</td>
          <td>20.309459</td>
          <td>0.005079</td>
          <td>22.502444</td>
          <td>0.011873</td>
          <td>21.221908</td>
          <td>0.009256</td>
          <td>19.398884</td>
          <td>24.627587</td>
          <td>23.089795</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.562542</td>
          <td>0.395721</td>
          <td>23.412984</td>
          <td>0.010171</td>
          <td>20.914816</td>
          <td>0.005086</td>
          <td>19.983186</td>
          <td>0.005049</td>
          <td>26.083030</td>
          <td>0.255201</td>
          <td>24.823283</td>
          <td>0.192257</td>
          <td>25.057373</td>
          <td>27.254548</td>
          <td>25.291235</td>
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
          <td>15.143606</td>
          <td>0.005002</td>
          <td>22.881727</td>
          <td>0.007476</td>
          <td>18.398107</td>
          <td>0.005004</td>
          <td>20.488196</td>
          <td>0.005104</td>
          <td>20.621371</td>
          <td>0.005413</td>
          <td>20.498581</td>
          <td>0.006470</td>
          <td>25.385269</td>
          <td>28.606679</td>
          <td>24.419141</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.883614</td>
          <td>0.005072</td>
          <td>25.757943</td>
          <td>0.072396</td>
          <td>23.328636</td>
          <td>0.008700</td>
          <td>21.744284</td>
          <td>0.005767</td>
          <td>14.405894</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.491173</td>
          <td>21.812410</td>
          <td>18.880427</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.164252</td>
          <td>0.021887</td>
          <td>20.850776</td>
          <td>0.005117</td>
          <td>24.942859</td>
          <td>0.030915</td>
          <td>22.450665</td>
          <td>0.007309</td>
          <td>25.163785</td>
          <td>0.117038</td>
          <td>21.357163</td>
          <td>0.010115</td>
          <td>24.091214</td>
          <td>23.163089</td>
          <td>23.181289</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.580875</td>
          <td>0.031231</td>
          <td>25.723570</td>
          <td>0.070231</td>
          <td>23.288012</td>
          <td>0.008494</td>
          <td>22.709173</td>
          <td>0.008368</td>
          <td>22.105865</td>
          <td>0.009052</td>
          <td>14.435163</td>
          <td>0.005001</td>
          <td>25.341520</td>
          <td>23.577715</td>
          <td>28.622641</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.683393</td>
          <td>0.194404</td>
          <td>26.466364</td>
          <td>0.134532</td>
          <td>20.218435</td>
          <td>0.005032</td>
          <td>26.781728</td>
          <td>0.248134</td>
          <td>21.122858</td>
          <td>0.005923</td>
          <td>24.205443</td>
          <td>0.113121</td>
          <td>24.706970</td>
          <td>21.233421</td>
          <td>20.396110</td>
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
          <td>18.939778</td>
          <td>19.072641</td>
          <td>26.163908</td>
          <td>20.553186</td>
          <td>21.799546</td>
          <td>21.114998</td>
          <td>24.814921</td>
          <td>0.029059</td>
          <td>22.729079</td>
          <td>0.008889</td>
          <td>27.469738</td>
          <td>0.465783</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.388917</td>
          <td>20.036402</td>
          <td>20.117551</td>
          <td>26.215399</td>
          <td>21.132268</td>
          <td>22.241236</td>
          <td>24.711970</td>
          <td>0.026541</td>
          <td>28.225669</td>
          <td>0.792407</td>
          <td>24.133708</td>
          <td>0.027053</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.276329</td>
          <td>18.860751</td>
          <td>25.106899</td>
          <td>28.275819</td>
          <td>20.681661</td>
          <td>18.706773</td>
          <td>25.622429</td>
          <td>0.059595</td>
          <td>19.618713</td>
          <td>0.005018</td>
          <td>24.040323</td>
          <td>0.024925</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.978888</td>
          <td>23.160220</td>
          <td>30.082883</td>
          <td>20.308177</td>
          <td>22.481479</td>
          <td>21.212916</td>
          <td>19.406954</td>
          <td>0.005004</td>
          <td>24.639962</td>
          <td>0.042381</td>
          <td>23.093621</td>
          <td>0.011423</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.325230</td>
          <td>23.412856</td>
          <td>20.921770</td>
          <td>19.978369</td>
          <td>27.239717</td>
          <td>24.703266</td>
          <td>24.987404</td>
          <td>0.033852</td>
          <td>27.510521</td>
          <td>0.480184</td>
          <td>25.356049</td>
          <td>0.080138</td>
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
          <td>15.145201</td>
          <td>22.872954</td>
          <td>18.393378</td>
          <td>20.489487</td>
          <td>20.622421</td>
          <td>20.489121</td>
          <td>25.406400</td>
          <td>0.049160</td>
          <td>28.502936</td>
          <td>0.944821</td>
          <td>24.456239</td>
          <td>0.035987</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.884553</td>
          <td>25.767245</td>
          <td>23.337443</td>
          <td>21.738545</td>
          <td>14.398790</td>
          <td>26.188011</td>
          <td>28.305633</td>
          <td>0.554007</td>
          <td>21.811465</td>
          <td>0.005915</td>
          <td>18.879339</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.158100</td>
          <td>20.846878</td>
          <td>24.948462</td>
          <td>22.433533</td>
          <td>25.192708</td>
          <td>21.361917</td>
          <td>24.092780</td>
          <td>0.015629</td>
          <td>23.169630</td>
          <td>0.012095</td>
          <td>23.170949</td>
          <td>0.012107</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.579910</td>
          <td>25.722199</td>
          <td>23.289484</td>
          <td>22.704111</td>
          <td>22.113844</td>
          <td>14.438278</td>
          <td>25.371758</td>
          <td>0.047665</td>
          <td>23.570623</td>
          <td>0.016669</td>
          <td>27.623307</td>
          <td>0.521851</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.602941</td>
          <td>26.179746</td>
          <td>20.217659</td>
          <td>26.744500</td>
          <td>21.119197</td>
          <td>24.263217</td>
          <td>24.688462</td>
          <td>0.025999</td>
          <td>21.232185</td>
          <td>0.005333</td>
          <td>20.392071</td>
          <td>0.005073</td>
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
          <td>18.939778</td>
          <td>19.072641</td>
          <td>26.163908</td>
          <td>20.553186</td>
          <td>21.799546</td>
          <td>21.114998</td>
          <td>24.591948</td>
          <td>0.252428</td>
          <td>22.769187</td>
          <td>0.043500</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.388917</td>
          <td>20.036402</td>
          <td>20.117551</td>
          <td>26.215399</td>
          <td>21.132268</td>
          <td>22.241236</td>
          <td>24.413624</td>
          <td>0.217778</td>
          <td>29.204160</td>
          <td>3.025782</td>
          <td>24.147681</td>
          <td>0.159828</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.276329</td>
          <td>18.860751</td>
          <td>25.106899</td>
          <td>28.275819</td>
          <td>20.681661</td>
          <td>18.706773</td>
          <td>25.151700</td>
          <td>0.394595</td>
          <td>19.627225</td>
          <td>0.005560</td>
          <td>24.132145</td>
          <td>0.157716</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.978888</td>
          <td>23.160220</td>
          <td>30.082883</td>
          <td>20.308177</td>
          <td>22.481479</td>
          <td>21.212916</td>
          <td>19.401346</td>
          <td>0.005536</td>
          <td>24.686519</td>
          <td>0.231390</td>
          <td>23.074864</td>
          <td>0.062442</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.325230</td>
          <td>23.412856</td>
          <td>20.921770</td>
          <td>19.978369</td>
          <td>27.239717</td>
          <td>24.703266</td>
          <td>25.159082</td>
          <td>0.396850</td>
          <td>25.925715</td>
          <td>0.603601</td>
          <td>24.475294</td>
          <td>0.210915</td>
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
          <td>15.145201</td>
          <td>22.872954</td>
          <td>18.393378</td>
          <td>20.489487</td>
          <td>20.622421</td>
          <td>20.489121</td>
          <td>25.685983</td>
          <td>0.586835</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.685491</td>
          <td>0.251092</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.884553</td>
          <td>25.767245</td>
          <td>23.337443</td>
          <td>21.738545</td>
          <td>14.398790</td>
          <td>26.188011</td>
          <td>28.182262</td>
          <td>2.277080</td>
          <td>21.801073</td>
          <td>0.018601</td>
          <td>18.878809</td>
          <td>0.005176</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.158100</td>
          <td>20.846878</td>
          <td>24.948462</td>
          <td>22.433533</td>
          <td>25.192708</td>
          <td>21.361917</td>
          <td>24.194061</td>
          <td>0.181053</td>
          <td>23.108300</td>
          <td>0.058850</td>
          <td>23.316761</td>
          <td>0.077399</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.579910</td>
          <td>25.722199</td>
          <td>23.289484</td>
          <td>22.704111</td>
          <td>22.113844</td>
          <td>14.438278</td>
          <td>24.870771</td>
          <td>0.316432</td>
          <td>23.600377</td>
          <td>0.091029</td>
          <td>25.324426</td>
          <td>0.417274</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.602941</td>
          <td>26.179746</td>
          <td>20.217659</td>
          <td>26.744500</td>
          <td>21.119197</td>
          <td>24.263217</td>
          <td>25.561535</td>
          <td>0.536598</td>
          <td>21.236423</td>
          <td>0.011795</td>
          <td>20.395346</td>
          <td>0.007365</td>
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

