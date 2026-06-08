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
    /home/runner/.cache/lephare/runs/20260608T131703


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
      File "/tmp/ipykernel_4206/2313627096.py", line 5, in <module>
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
          <td>23.264666</td>
          <td>18.787931</td>
          <td>19.099515</td>
          <td>29.831094</td>
          <td>22.834884</td>
          <td>24.477505</td>
          <td>25.194382</td>
          <td>20.337620</td>
          <td>24.549454</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.012257</td>
          <td>27.778744</td>
          <td>23.675052</td>
          <td>25.583585</td>
          <td>23.084580</td>
          <td>21.758897</td>
          <td>22.618910</td>
          <td>19.933131</td>
          <td>21.878075</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.292218</td>
          <td>22.737114</td>
          <td>25.509822</td>
          <td>22.418943</td>
          <td>23.215301</td>
          <td>21.646389</td>
          <td>28.062050</td>
          <td>20.865957</td>
          <td>28.184333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.250284</td>
          <td>28.419214</td>
          <td>27.242522</td>
          <td>23.908919</td>
          <td>18.442204</td>
          <td>26.480220</td>
          <td>20.517774</td>
          <td>21.787736</td>
          <td>21.441008</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.261431</td>
          <td>21.282729</td>
          <td>20.668091</td>
          <td>23.434570</td>
          <td>20.374939</td>
          <td>23.678006</td>
          <td>22.240018</td>
          <td>21.521298</td>
          <td>26.257025</td>
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
          <td>23.484727</td>
          <td>21.757996</td>
          <td>25.266357</td>
          <td>23.233763</td>
          <td>26.969621</td>
          <td>25.377137</td>
          <td>23.401180</td>
          <td>24.848924</td>
          <td>26.163593</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.205074</td>
          <td>23.101688</td>
          <td>20.864030</td>
          <td>19.688462</td>
          <td>25.356431</td>
          <td>25.167394</td>
          <td>26.894666</td>
          <td>27.866200</td>
          <td>22.496721</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.921286</td>
          <td>25.574388</td>
          <td>16.108822</td>
          <td>26.141817</td>
          <td>26.481981</td>
          <td>22.064149</td>
          <td>19.191864</td>
          <td>25.929276</td>
          <td>25.372825</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.029955</td>
          <td>22.522634</td>
          <td>19.347139</td>
          <td>15.093078</td>
          <td>25.338210</td>
          <td>18.898560</td>
          <td>19.459682</td>
          <td>23.105256</td>
          <td>23.215118</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.555363</td>
          <td>20.989810</td>
          <td>18.661879</td>
          <td>21.294863</td>
          <td>21.995244</td>
          <td>26.073047</td>
          <td>18.054942</td>
          <td>24.164361</td>
          <td>15.024536</td>
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
          <td>23.259813</td>
          <td>0.023720</td>
          <td>18.784993</td>
          <td>0.005009</td>
          <td>19.106627</td>
          <td>0.005008</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.841460</td>
          <td>0.015449</td>
          <td>24.475142</td>
          <td>0.142899</td>
          <td>25.194382</td>
          <td>20.337620</td>
          <td>24.549454</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.008825</td>
          <td>0.006093</td>
          <td>27.283040</td>
          <td>0.267509</td>
          <td>23.672812</td>
          <td>0.010907</td>
          <td>25.507598</td>
          <td>0.083217</td>
          <td>23.073244</td>
          <td>0.018701</td>
          <td>21.740865</td>
          <td>0.013383</td>
          <td>22.618910</td>
          <td>19.933131</td>
          <td>21.878075</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.284763</td>
          <td>0.005038</td>
          <td>22.749380</td>
          <td>0.007040</td>
          <td>25.594150</td>
          <td>0.055036</td>
          <td>22.420205</td>
          <td>0.007206</td>
          <td>23.207420</td>
          <td>0.020948</td>
          <td>21.641187</td>
          <td>0.012401</td>
          <td>28.062050</td>
          <td>20.865957</td>
          <td>28.184333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.138804</td>
          <td>0.050870</td>
          <td>30.500914</td>
          <td>2.016415</td>
          <td>27.062902</td>
          <td>0.197818</td>
          <td>23.925244</td>
          <td>0.020670</td>
          <td>18.445393</td>
          <td>0.005017</td>
          <td>27.112336</td>
          <td>1.034613</td>
          <td>20.517774</td>
          <td>21.787736</td>
          <td>21.441008</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.265147</td>
          <td>0.005392</td>
          <td>21.287991</td>
          <td>0.005221</td>
          <td>20.668579</td>
          <td>0.005060</td>
          <td>23.450594</td>
          <td>0.013985</td>
          <td>20.380480</td>
          <td>0.005281</td>
          <td>23.775099</td>
          <td>0.077530</td>
          <td>22.240018</td>
          <td>21.521298</td>
          <td>26.257025</td>
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
          <td>23.509892</td>
          <td>0.029373</td>
          <td>21.759491</td>
          <td>0.005451</td>
          <td>25.255453</td>
          <td>0.040746</td>
          <td>23.245605</td>
          <td>0.011953</td>
          <td>27.846758</td>
          <td>0.924848</td>
          <td>25.108645</td>
          <td>0.243894</td>
          <td>23.401180</td>
          <td>24.848924</td>
          <td>26.163593</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.208853</td>
          <td>0.006446</td>
          <td>23.100877</td>
          <td>0.008383</td>
          <td>20.872079</td>
          <td>0.005081</td>
          <td>19.689424</td>
          <td>0.005032</td>
          <td>25.458920</td>
          <td>0.151047</td>
          <td>25.103690</td>
          <td>0.242900</td>
          <td>26.894666</td>
          <td>27.866200</td>
          <td>22.496721</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.920426</td>
          <td>0.008816</td>
          <td>25.598806</td>
          <td>0.062895</td>
          <td>16.104287</td>
          <td>0.005000</td>
          <td>26.367779</td>
          <td>0.175520</td>
          <td>26.302992</td>
          <td>0.305061</td>
          <td>22.061598</td>
          <td>0.017317</td>
          <td>19.191864</td>
          <td>25.929276</td>
          <td>25.372825</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.069642</td>
          <td>0.047869</td>
          <td>22.521431</td>
          <td>0.006452</td>
          <td>19.333963</td>
          <td>0.005010</td>
          <td>15.091728</td>
          <td>0.005000</td>
          <td>25.178894</td>
          <td>0.118586</td>
          <td>18.896272</td>
          <td>0.005116</td>
          <td>19.459682</td>
          <td>23.105256</td>
          <td>23.215118</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.558381</td>
          <td>0.007346</td>
          <td>20.988776</td>
          <td>0.005143</td>
          <td>18.673345</td>
          <td>0.005005</td>
          <td>21.298812</td>
          <td>0.005374</td>
          <td>22.003149</td>
          <td>0.008509</td>
          <td>25.856088</td>
          <td>0.441021</td>
          <td>18.054942</td>
          <td>24.164361</td>
          <td>15.024536</td>
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
          <td>23.264666</td>
          <td>18.787931</td>
          <td>19.099515</td>
          <td>29.831094</td>
          <td>22.834884</td>
          <td>24.477505</td>
          <td>25.211996</td>
          <td>0.041338</td>
          <td>20.333316</td>
          <td>0.005065</td>
          <td>24.596740</td>
          <td>0.040780</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.012257</td>
          <td>27.778744</td>
          <td>23.675052</td>
          <td>25.583585</td>
          <td>23.084580</td>
          <td>21.758897</td>
          <td>22.614720</td>
          <td>0.006286</td>
          <td>19.927625</td>
          <td>0.005031</td>
          <td>21.875541</td>
          <td>0.006020</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.292218</td>
          <td>22.737114</td>
          <td>25.509822</td>
          <td>22.418943</td>
          <td>23.215301</td>
          <td>21.646389</td>
          <td>27.572776</td>
          <td>0.316940</td>
          <td>20.875977</td>
          <td>0.005175</td>
          <td>28.820815</td>
          <td>1.140662</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.250284</td>
          <td>28.419214</td>
          <td>27.242522</td>
          <td>23.908919</td>
          <td>18.442204</td>
          <td>26.480220</td>
          <td>20.525704</td>
          <td>0.005031</td>
          <td>21.785522</td>
          <td>0.005875</td>
          <td>21.446064</td>
          <td>0.005486</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.261431</td>
          <td>21.282729</td>
          <td>20.668091</td>
          <td>23.434570</td>
          <td>20.374939</td>
          <td>23.678006</td>
          <td>22.247089</td>
          <td>0.005690</td>
          <td>21.513734</td>
          <td>0.005547</td>
          <td>26.229304</td>
          <td>0.171361</td>
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
          <td>23.484727</td>
          <td>21.757996</td>
          <td>25.266357</td>
          <td>23.233763</td>
          <td>26.969621</td>
          <td>25.377137</td>
          <td>23.411397</td>
          <td>0.009372</td>
          <td>24.798388</td>
          <td>0.048810</td>
          <td>26.176183</td>
          <td>0.163771</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.205074</td>
          <td>23.101688</td>
          <td>20.864030</td>
          <td>19.688462</td>
          <td>25.356431</td>
          <td>25.167394</td>
          <td>26.964896</td>
          <td>0.192231</td>
          <td>27.171551</td>
          <td>0.370796</td>
          <td>22.490769</td>
          <td>0.007736</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.921286</td>
          <td>25.574388</td>
          <td>16.108822</td>
          <td>26.141817</td>
          <td>26.481981</td>
          <td>22.064149</td>
          <td>19.190088</td>
          <td>0.005003</td>
          <td>25.755525</td>
          <td>0.113860</td>
          <td>25.471658</td>
          <td>0.088753</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.029955</td>
          <td>22.522634</td>
          <td>19.347139</td>
          <td>15.093078</td>
          <td>25.338210</td>
          <td>18.898560</td>
          <td>19.443744</td>
          <td>0.005004</td>
          <td>23.107841</td>
          <td>0.011545</td>
          <td>23.224451</td>
          <td>0.012614</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.555363</td>
          <td>20.989810</td>
          <td>18.661879</td>
          <td>21.294863</td>
          <td>21.995244</td>
          <td>26.073047</td>
          <td>18.048955</td>
          <td>0.005000</td>
          <td>24.156931</td>
          <td>0.027611</td>
          <td>15.034683</td>
          <td>0.005000</td>
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
          <td>23.264666</td>
          <td>18.787931</td>
          <td>19.099515</td>
          <td>29.831094</td>
          <td>22.834884</td>
          <td>24.477505</td>
          <td>24.831039</td>
          <td>0.306526</td>
          <td>20.344942</td>
          <td>0.006868</td>
          <td>24.334930</td>
          <td>0.187427</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.012257</td>
          <td>27.778744</td>
          <td>23.675052</td>
          <td>25.583585</td>
          <td>23.084580</td>
          <td>21.758897</td>
          <td>22.611027</td>
          <td>0.045153</td>
          <td>19.923696</td>
          <td>0.005934</td>
          <td>21.877602</td>
          <td>0.021637</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.292218</td>
          <td>22.737114</td>
          <td>25.509822</td>
          <td>22.418943</td>
          <td>23.215301</td>
          <td>21.646389</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.864264</td>
          <td>0.009090</td>
          <td>25.618915</td>
          <td>0.520177</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.250284</td>
          <td>28.419214</td>
          <td>27.242522</td>
          <td>23.908919</td>
          <td>18.442204</td>
          <td>26.480220</td>
          <td>20.513710</td>
          <td>0.008288</td>
          <td>21.831902</td>
          <td>0.019095</td>
          <td>21.446386</td>
          <td>0.015047</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.261431</td>
          <td>21.282729</td>
          <td>20.668091</td>
          <td>23.434570</td>
          <td>20.374939</td>
          <td>23.678006</td>
          <td>22.234926</td>
          <td>0.032313</td>
          <td>21.510255</td>
          <td>0.014612</td>
          <td>25.401897</td>
          <td>0.442597</td>
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
          <td>23.484727</td>
          <td>21.757996</td>
          <td>25.266357</td>
          <td>23.233763</td>
          <td>26.969621</td>
          <td>25.377137</td>
          <td>23.404888</td>
          <td>0.091391</td>
          <td>25.596283</td>
          <td>0.475116</td>
          <td>26.883601</td>
          <td>1.181900</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.205074</td>
          <td>23.101688</td>
          <td>20.864030</td>
          <td>19.688462</td>
          <td>25.356431</td>
          <td>25.167394</td>
          <td>29.699646</td>
          <td>3.689155</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.457933</td>
          <td>0.036041</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.921286</td>
          <td>25.574388</td>
          <td>16.108822</td>
          <td>26.141817</td>
          <td>26.481981</td>
          <td>22.064149</td>
          <td>19.200926</td>
          <td>0.005376</td>
          <td>25.590193</td>
          <td>0.472962</td>
          <td>25.641438</td>
          <td>0.528806</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.029955</td>
          <td>22.522634</td>
          <td>19.347139</td>
          <td>15.093078</td>
          <td>25.338210</td>
          <td>18.898560</td>
          <td>19.447715</td>
          <td>0.005581</td>
          <td>23.076969</td>
          <td>0.057231</td>
          <td>23.170137</td>
          <td>0.067961</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.555363</td>
          <td>20.989810</td>
          <td>18.661879</td>
          <td>21.294863</td>
          <td>21.995244</td>
          <td>26.073047</td>
          <td>18.055853</td>
          <td>0.005047</td>
          <td>24.194159</td>
          <td>0.152662</td>
          <td>15.019206</td>
          <td>0.005000</td>
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

