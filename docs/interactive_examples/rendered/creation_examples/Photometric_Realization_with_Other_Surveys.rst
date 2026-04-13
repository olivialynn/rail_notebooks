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
    /home/runner/.cache/lephare/runs/20260413T121611


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
      File "/tmp/ipykernel_5901/2313627096.py", line 5, in <module>
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
          <td>25.897894</td>
          <td>23.339046</td>
          <td>21.623693</td>
          <td>24.904503</td>
          <td>22.118124</td>
          <td>19.142451</td>
          <td>23.604477</td>
          <td>26.805181</td>
          <td>25.207503</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.084924</td>
          <td>21.159183</td>
          <td>23.392202</td>
          <td>21.833140</td>
          <td>28.422065</td>
          <td>26.645546</td>
          <td>13.461125</td>
          <td>24.763819</td>
          <td>22.103713</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.052481</td>
          <td>21.952178</td>
          <td>26.590488</td>
          <td>25.160505</td>
          <td>16.352804</td>
          <td>25.917952</td>
          <td>23.896386</td>
          <td>23.361532</td>
          <td>24.377816</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.619152</td>
          <td>23.470968</td>
          <td>25.014827</td>
          <td>21.212693</td>
          <td>22.038772</td>
          <td>23.732321</td>
          <td>23.510911</td>
          <td>24.512017</td>
          <td>25.373988</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.835192</td>
          <td>21.867616</td>
          <td>20.511714</td>
          <td>23.952375</td>
          <td>17.679979</td>
          <td>26.721318</td>
          <td>23.473195</td>
          <td>19.330594</td>
          <td>20.944897</td>
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
          <td>23.390023</td>
          <td>26.791017</td>
          <td>19.570421</td>
          <td>25.241961</td>
          <td>24.268228</td>
          <td>27.976257</td>
          <td>21.939831</td>
          <td>29.143418</td>
          <td>17.464569</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.911386</td>
          <td>21.815429</td>
          <td>24.878357</td>
          <td>25.730735</td>
          <td>21.251867</td>
          <td>26.542176</td>
          <td>19.060214</td>
          <td>21.369344</td>
          <td>18.359437</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.458356</td>
          <td>20.984413</td>
          <td>25.340463</td>
          <td>20.543045</td>
          <td>21.435707</td>
          <td>17.463520</td>
          <td>15.823049</td>
          <td>24.395134</td>
          <td>23.414510</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.130299</td>
          <td>15.947171</td>
          <td>30.237984</td>
          <td>23.404187</td>
          <td>23.647610</td>
          <td>24.426574</td>
          <td>21.390735</td>
          <td>26.705496</td>
          <td>20.665431</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.338789</td>
          <td>24.521082</td>
          <td>20.616191</td>
          <td>15.628695</td>
          <td>20.725359</td>
          <td>24.529839</td>
          <td>18.898991</td>
          <td>24.431981</td>
          <td>21.563523</td>
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
          <td>25.759992</td>
          <td>0.207288</td>
          <td>23.326296</td>
          <td>0.009607</td>
          <td>21.623517</td>
          <td>0.005260</td>
          <td>24.852086</td>
          <td>0.046552</td>
          <td>22.112628</td>
          <td>0.009090</td>
          <td>19.142646</td>
          <td>0.005170</td>
          <td>23.604477</td>
          <td>26.805181</td>
          <td>25.207503</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.109683</td>
          <td>0.049584</td>
          <td>21.157323</td>
          <td>0.005182</td>
          <td>23.379799</td>
          <td>0.008975</td>
          <td>21.832126</td>
          <td>0.005883</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.200549</td>
          <td>1.089646</td>
          <td>13.461125</td>
          <td>24.763819</td>
          <td>22.103713</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.037649</td>
          <td>0.019700</td>
          <td>21.950655</td>
          <td>0.005605</td>
          <td>26.652032</td>
          <td>0.139388</td>
          <td>25.095387</td>
          <td>0.057778</td>
          <td>16.350434</td>
          <td>0.005002</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.896386</td>
          <td>23.361532</td>
          <td>24.377816</td>
        </tr>
        <tr>
          <th>3</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.476503</td>
          <td>0.010620</td>
          <td>25.016184</td>
          <td>0.032977</td>
          <td>21.219700</td>
          <td>0.005329</td>
          <td>22.041713</td>
          <td>0.008705</td>
          <td>23.675191</td>
          <td>0.070977</td>
          <td>23.510911</td>
          <td>24.512017</td>
          <td>25.373988</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.339240</td>
          <td>0.696059</td>
          <td>21.869097</td>
          <td>0.005534</td>
          <td>20.509719</td>
          <td>0.005047</td>
          <td>23.946317</td>
          <td>0.021045</td>
          <td>17.688032</td>
          <td>0.005007</td>
          <td>25.907690</td>
          <td>0.458513</td>
          <td>23.473195</td>
          <td>19.330594</td>
          <td>20.944897</td>
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
          <td>23.395878</td>
          <td>0.026632</td>
          <td>26.692298</td>
          <td>0.163316</td>
          <td>19.563867</td>
          <td>0.005013</td>
          <td>25.317185</td>
          <td>0.070332</td>
          <td>24.270385</td>
          <td>0.053251</td>
          <td>26.368122</td>
          <td>0.639926</td>
          <td>21.939831</td>
          <td>29.143418</td>
          <td>17.464569</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.905036</td>
          <td>0.008740</td>
          <td>21.817031</td>
          <td>0.005493</td>
          <td>24.883016</td>
          <td>0.029333</td>
          <td>25.709950</td>
          <td>0.099419</td>
          <td>21.246557</td>
          <td>0.006123</td>
          <td>27.132489</td>
          <td>1.047038</td>
          <td>19.060214</td>
          <td>21.369344</td>
          <td>18.359437</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.464094</td>
          <td>0.161483</td>
          <td>20.987171</td>
          <td>0.005143</td>
          <td>25.413004</td>
          <td>0.046859</td>
          <td>20.551455</td>
          <td>0.005114</td>
          <td>21.440922</td>
          <td>0.006522</td>
          <td>17.456189</td>
          <td>0.005016</td>
          <td>15.823049</td>
          <td>24.395134</td>
          <td>23.414510</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.130086</td>
          <td>0.005032</td>
          <td>15.949419</td>
          <td>0.005001</td>
          <td>31.008515</td>
          <td>2.330995</td>
          <td>23.395978</td>
          <td>0.013401</td>
          <td>23.671987</td>
          <td>0.031350</td>
          <td>24.672581</td>
          <td>0.169216</td>
          <td>21.390735</td>
          <td>26.705496</td>
          <td>20.665431</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.343305</td>
          <td>0.005040</td>
          <td>24.514139</td>
          <td>0.024227</td>
          <td>20.611139</td>
          <td>0.005055</td>
          <td>15.639053</td>
          <td>0.005000</td>
          <td>20.728815</td>
          <td>0.005491</td>
          <td>24.623741</td>
          <td>0.162316</td>
          <td>18.898991</td>
          <td>24.431981</td>
          <td>21.563523</td>
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
          <td>25.897894</td>
          <td>23.339046</td>
          <td>21.623693</td>
          <td>24.904503</td>
          <td>22.118124</td>
          <td>19.142451</td>
          <td>23.590304</td>
          <td>0.010596</td>
          <td>26.554652</td>
          <td>0.225348</td>
          <td>25.257017</td>
          <td>0.073408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.084924</td>
          <td>21.159183</td>
          <td>23.392202</td>
          <td>21.833140</td>
          <td>28.422065</td>
          <td>26.645546</td>
          <td>13.461300</td>
          <td>0.005000</td>
          <td>24.806167</td>
          <td>0.049150</td>
          <td>22.101686</td>
          <td>0.006484</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.052481</td>
          <td>21.952178</td>
          <td>26.590488</td>
          <td>25.160505</td>
          <td>16.352804</td>
          <td>25.917952</td>
          <td>23.898844</td>
          <td>0.013368</td>
          <td>23.350873</td>
          <td>0.013931</td>
          <td>24.384752</td>
          <td>0.033773</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.619152</td>
          <td>23.470968</td>
          <td>25.014827</td>
          <td>21.212693</td>
          <td>22.038772</td>
          <td>23.732321</td>
          <td>23.499423</td>
          <td>0.009943</td>
          <td>24.517993</td>
          <td>0.038019</td>
          <td>25.214387</td>
          <td>0.070684</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.835192</td>
          <td>21.867616</td>
          <td>20.511714</td>
          <td>23.952375</td>
          <td>17.679979</td>
          <td>26.721318</td>
          <td>23.473365</td>
          <td>0.009768</td>
          <td>19.327056</td>
          <td>0.005010</td>
          <td>20.944475</td>
          <td>0.005198</td>
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
          <td>23.390023</td>
          <td>26.791017</td>
          <td>19.570421</td>
          <td>25.241961</td>
          <td>24.268228</td>
          <td>27.976257</td>
          <td>21.934822</td>
          <td>0.005399</td>
          <td>28.677117</td>
          <td>1.049410</td>
          <td>17.465245</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.911386</td>
          <td>21.815429</td>
          <td>24.878357</td>
          <td>25.730735</td>
          <td>21.251867</td>
          <td>26.542176</td>
          <td>19.061126</td>
          <td>0.005002</td>
          <td>21.378016</td>
          <td>0.005431</td>
          <td>18.367852</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.458356</td>
          <td>20.984413</td>
          <td>25.340463</td>
          <td>20.543045</td>
          <td>21.435707</td>
          <td>17.463520</td>
          <td>15.822924</td>
          <td>0.005000</td>
          <td>24.375652</td>
          <td>0.033501</td>
          <td>23.429093</td>
          <td>0.014837</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.130299</td>
          <td>15.947171</td>
          <td>30.237984</td>
          <td>23.404187</td>
          <td>23.647610</td>
          <td>24.426574</td>
          <td>21.395068</td>
          <td>0.005151</td>
          <td>26.413360</td>
          <td>0.200239</td>
          <td>20.672167</td>
          <td>0.005121</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.338789</td>
          <td>24.521082</td>
          <td>20.616191</td>
          <td>15.628695</td>
          <td>20.725359</td>
          <td>24.529839</td>
          <td>18.894480</td>
          <td>0.005002</td>
          <td>24.493805</td>
          <td>0.037209</td>
          <td>21.576980</td>
          <td>0.005611</td>
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
          <td>25.897894</td>
          <td>23.339046</td>
          <td>21.623693</td>
          <td>24.904503</td>
          <td>22.118124</td>
          <td>19.142451</td>
          <td>23.535577</td>
          <td>0.102516</td>
          <td>26.692371</td>
          <td>0.997686</td>
          <td>25.674819</td>
          <td>0.541797</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.084924</td>
          <td>21.159183</td>
          <td>23.392202</td>
          <td>21.833140</td>
          <td>28.422065</td>
          <td>26.645546</td>
          <td>13.457769</td>
          <td>0.005000</td>
          <td>24.907766</td>
          <td>0.277484</td>
          <td>22.068368</td>
          <td>0.025545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.052481</td>
          <td>21.952178</td>
          <td>26.590488</td>
          <td>25.160505</td>
          <td>16.352804</td>
          <td>25.917952</td>
          <td>24.074558</td>
          <td>0.163544</td>
          <td>23.398169</td>
          <td>0.076135</td>
          <td>24.430566</td>
          <td>0.203154</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.619152</td>
          <td>23.470968</td>
          <td>25.014827</td>
          <td>21.212693</td>
          <td>22.038772</td>
          <td>23.732321</td>
          <td>23.618458</td>
          <td>0.110233</td>
          <td>24.212972</td>
          <td>0.155146</td>
          <td>25.170731</td>
          <td>0.370559</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.835192</td>
          <td>21.867616</td>
          <td>20.511714</td>
          <td>23.952375</td>
          <td>17.679979</td>
          <td>26.721318</td>
          <td>23.447483</td>
          <td>0.094883</td>
          <td>19.336141</td>
          <td>0.005335</td>
          <td>20.946055</td>
          <td>0.010269</td>
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
          <td>23.390023</td>
          <td>26.791017</td>
          <td>19.570421</td>
          <td>25.241961</td>
          <td>24.268228</td>
          <td>27.976257</td>
          <td>21.934082</td>
          <td>0.024790</td>
          <td>26.856699</td>
          <td>1.099403</td>
          <td>17.458894</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.911386</td>
          <td>21.815429</td>
          <td>24.878357</td>
          <td>25.730735</td>
          <td>21.251867</td>
          <td>26.542176</td>
          <td>19.060474</td>
          <td>0.005293</td>
          <td>21.350430</td>
          <td>0.012871</td>
          <td>18.356108</td>
          <td>0.005068</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.458356</td>
          <td>20.984413</td>
          <td>25.340463</td>
          <td>20.543045</td>
          <td>21.435707</td>
          <td>17.463520</td>
          <td>15.824723</td>
          <td>0.005001</td>
          <td>24.502130</td>
          <td>0.198357</td>
          <td>23.400850</td>
          <td>0.083376</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.130299</td>
          <td>15.947171</td>
          <td>30.237984</td>
          <td>23.404187</td>
          <td>23.647610</td>
          <td>24.426574</td>
          <td>21.380268</td>
          <td>0.015470</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.655534</td>
          <td>0.008496</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.338789</td>
          <td>24.521082</td>
          <td>20.616191</td>
          <td>15.628695</td>
          <td>20.725359</td>
          <td>24.529839</td>
          <td>18.903157</td>
          <td>0.005221</td>
          <td>24.394641</td>
          <td>0.181142</td>
          <td>21.549130</td>
          <td>0.016374</td>
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

