Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: Feb 9, 2026

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``Quick_Start_in_Creation.ipynb``

If you’re interested in running this in pipeline mode, see
```01_Photometric_Realization.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/creation_examples/01_Photometric_Realization.ipynb>`__
in the ``pipeline_examples/creation_examples/`` folder.

.. code:: ipython3

    import os
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pzflow
    import rail.interactive as ri


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
      File "/tmp/ipykernel_4923/696671147.py", line 6, in <module>
        import rail.interactive as ri
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


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )

“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

.. code:: ipython3

    n_samples = int(1e5)
    
    samples_truth = ri.creation.engines.flowEngine.flow_creator(
        n_samples=n_samples, model=flow_file, seed=0
    )


.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, FlowCreator


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator


.. code:: ipython3

    samples_truth["output"]["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth["output"]["minor"] = samples_truth["output"]["major"] * b_to_a
    
    samples_truth["output"]




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
          <th>redshift</th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.400313</td>
          <td>27.358033</td>
          <td>26.623911</td>
          <td>25.686214</td>
          <td>25.423050</td>
          <td>25.204436</td>
          <td>25.080073</td>
          <td>0.048953</td>
          <td>0.038597</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.983561</td>
          <td>27.248891</td>
          <td>26.808618</td>
          <td>26.391002</td>
          <td>26.103835</td>
          <td>25.704334</td>
          <td>25.487085</td>
          <td>0.005571</td>
          <td>0.005506</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.633238</td>
          <td>27.760765</td>
          <td>27.373345</td>
          <td>26.985904</td>
          <td>26.473378</td>
          <td>26.110453</td>
          <td>25.726033</td>
          <td>0.038664</td>
          <td>0.031715</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.593216</td>
          <td>27.390467</td>
          <td>26.602495</td>
          <td>25.782788</td>
          <td>25.323484</td>
          <td>25.215736</td>
          <td>25.061188</td>
          <td>0.014732</td>
          <td>0.011283</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.418184</td>
          <td>27.345716</td>
          <td>26.187644</td>
          <td>25.745173</td>
          <td>24.973738</td>
          <td>24.482849</td>
          <td>23.700091</td>
          <td>0.007457</td>
          <td>0.006633</td>
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
          <th>99995</th>
          <td>1.151420</td>
          <td>28.721063</td>
          <td>27.912678</td>
          <td>26.896495</td>
          <td>26.079070</td>
          <td>25.199434</td>
          <td>24.758864</td>
          <td>0.010310</td>
          <td>0.006117</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.359546</td>
          <td>26.043028</td>
          <td>25.756442</td>
          <td>25.520627</td>
          <td>25.080964</td>
          <td>24.695030</td>
          <td>24.119693</td>
          <td>0.067335</td>
          <td>0.052115</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.595311</td>
          <td>27.925256</td>
          <td>27.009046</td>
          <td>26.353701</td>
          <td>25.574312</td>
          <td>25.125290</td>
          <td>24.684619</td>
          <td>0.057751</td>
          <td>0.038064</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.849984</td>
          <td>27.391571</td>
          <td>27.111400</td>
          <td>26.471609</td>
          <td>25.585453</td>
          <td>25.270681</td>
          <td>25.192526</td>
          <td>0.071331</td>
          <td>0.048684</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>0.693812</td>
          <td>27.993451</td>
          <td>26.319147</td>
          <td>24.834481</td>
          <td>23.619452</td>
          <td>23.163058</td>
          <td>22.814961</td>
          <td>0.069849</td>
          <td>0.064400</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 9 columns</p>
    </div>



LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    samples_w_errs = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=samples_truth["output"]
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel


For extended sources:

.. code:: ipython3

    samples_w_errs_auto = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=samples_truth["output"], extendedType="auto"
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel


.. code:: ipython3

    samples_w_errs_gaap = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=samples_truth["output"], extendedType="gaap"
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel


Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs["output"][band].to_numpy()
        errs = samples_w_errs["output"][band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        # ax.plot(mags, errs, label=band)
    
        # plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(
            samples_w_errs_gaap["output"][band].to_numpy(),
            samples_w_errs_gaap["output"][band + "_err"].to_numpy(),
            s=5,
            marker=".",
            color="C0",
            alpha=0.8,
            label="GAAP",
        )
    
        ax.plot(mags, errs, color="C3", label="Point source")
    
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band + " Band Magnitude (AB)", ylabel="Error (mags)")



.. image:: Photometric_Realization_files/Photometric_Realization_14_0.png


.. code:: ipython3

    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs["output"][band].to_numpy()
        errs = samples_w_errs["output"][band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        # ax.plot(mags, errs, label=band)
    
        # plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(
            samples_w_errs_auto["output"][band].to_numpy(),
            samples_w_errs_auto["output"][band + "_err"].to_numpy(),
            s=5,
            marker=".",
            color="C0",
            alpha=0.8,
            label="AUTO",
        )
    
        ax.plot(mags, errs, color="C3", label="Point source")
    
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band + " Band Magnitude (AB)", ylabel="Error (mags)")



.. image:: Photometric_Realization_files/Photometric_Realization_15_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
