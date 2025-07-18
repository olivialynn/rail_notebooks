Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f27c531fca0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.016082  0.009486  
    1      25.391064  0.078366  0.056858  
    2      24.304707  0.217427  0.193605  
    3      25.291103  0.064558  0.039330  
    4      25.096743  0.175365  0.138129  
    ...          ...       ...       ...  
    99995  24.737946  0.061324  0.038267  
    99996  24.224169  0.003806  0.002306  
    99997  25.613836  0.054374  0.046719  
    99998  25.274899  0.266018  0.231582  
    99999  25.699642  0.002041  0.001876  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


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

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.529834</td>
          <td>0.385851</td>
          <td>26.509708</td>
          <td>0.139655</td>
          <td>26.181632</td>
          <td>0.092526</td>
          <td>25.107117</td>
          <td>0.058382</td>
          <td>24.732340</td>
          <td>0.080173</td>
          <td>23.940947</td>
          <td>0.089733</td>
          <td>0.016082</td>
          <td>0.009486</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.531034</td>
          <td>0.326628</td>
          <td>26.517179</td>
          <td>0.124040</td>
          <td>26.563580</td>
          <td>0.207033</td>
          <td>26.097796</td>
          <td>0.258308</td>
          <td>26.619375</td>
          <td>0.759013</td>
          <td>0.078366</td>
          <td>0.056858</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.866629</td>
          <td>0.780106</td>
          <td>26.183760</td>
          <td>0.150002</td>
          <td>25.040309</td>
          <td>0.105087</td>
          <td>24.308092</td>
          <td>0.123685</td>
          <td>0.217427</td>
          <td>0.193605</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.970419</td>
          <td>0.458811</td>
          <td>27.555810</td>
          <td>0.296919</td>
          <td>26.075252</td>
          <td>0.136625</td>
          <td>25.839785</td>
          <td>0.208618</td>
          <td>25.439559</td>
          <td>0.318989</td>
          <td>0.064558</td>
          <td>0.039330</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.144641</td>
          <td>0.284429</td>
          <td>26.107909</td>
          <td>0.098496</td>
          <td>25.924803</td>
          <td>0.073778</td>
          <td>25.552788</td>
          <td>0.086597</td>
          <td>25.482825</td>
          <td>0.154174</td>
          <td>25.325737</td>
          <td>0.291148</td>
          <td>0.175365</td>
          <td>0.138129</td>
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
          <th>99995</th>
          <td>0.389450</td>
          <td>27.185475</td>
          <td>0.626044</td>
          <td>26.382973</td>
          <td>0.125170</td>
          <td>25.396420</td>
          <td>0.046174</td>
          <td>24.994953</td>
          <td>0.052849</td>
          <td>24.745436</td>
          <td>0.081105</td>
          <td>24.851939</td>
          <td>0.196952</td>
          <td>0.061324</td>
          <td>0.038267</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.078731</td>
          <td>0.226116</td>
          <td>26.037140</td>
          <td>0.081474</td>
          <td>25.142660</td>
          <td>0.060253</td>
          <td>24.884489</td>
          <td>0.091669</td>
          <td>24.129426</td>
          <td>0.105859</td>
          <td>0.003806</td>
          <td>0.002306</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.327504</td>
          <td>0.329281</td>
          <td>26.691076</td>
          <td>0.163146</td>
          <td>26.434301</td>
          <td>0.115418</td>
          <td>26.539598</td>
          <td>0.202913</td>
          <td>25.701562</td>
          <td>0.185722</td>
          <td>26.188783</td>
          <td>0.563715</td>
          <td>0.054374</td>
          <td>0.046719</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.947597</td>
          <td>0.528240</td>
          <td>26.279801</td>
          <td>0.114443</td>
          <td>26.231842</td>
          <td>0.096696</td>
          <td>25.703904</td>
          <td>0.098894</td>
          <td>25.432695</td>
          <td>0.147684</td>
          <td>25.280739</td>
          <td>0.280739</td>
          <td>0.266018</td>
          <td>0.231582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.248281</td>
          <td>1.219548</td>
          <td>26.715826</td>
          <td>0.166624</td>
          <td>26.703921</td>
          <td>0.145757</td>
          <td>26.246906</td>
          <td>0.158341</td>
          <td>25.972667</td>
          <td>0.233018</td>
          <td>25.422935</td>
          <td>0.314785</td>
          <td>0.002041</td>
          <td>0.001876</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.114162</td>
          <td>0.653913</td>
          <td>26.770839</td>
          <td>0.200405</td>
          <td>26.059300</td>
          <td>0.097732</td>
          <td>25.139030</td>
          <td>0.071242</td>
          <td>24.688519</td>
          <td>0.090754</td>
          <td>23.970670</td>
          <td>0.108867</td>
          <td>0.016082</td>
          <td>0.009486</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.274181</td>
          <td>2.129639</td>
          <td>27.361027</td>
          <td>0.328887</td>
          <td>26.587412</td>
          <td>0.156810</td>
          <td>26.149019</td>
          <td>0.174237</td>
          <td>25.726032</td>
          <td>0.224445</td>
          <td>25.919747</td>
          <td>0.539471</td>
          <td>0.078366</td>
          <td>0.056858</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.247955</td>
          <td>0.771762</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.126669</td>
          <td>0.588067</td>
          <td>25.925936</td>
          <td>0.160884</td>
          <td>24.936217</td>
          <td>0.127616</td>
          <td>24.391439</td>
          <td>0.177439</td>
          <td>0.217427</td>
          <td>0.193605</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.267888</td>
          <td>1.330938</td>
          <td>28.900949</td>
          <td>0.973647</td>
          <td>27.277806</td>
          <td>0.277574</td>
          <td>25.840755</td>
          <td>0.132963</td>
          <td>25.062494</td>
          <td>0.126927</td>
          <td>25.261762</td>
          <td>0.324967</td>
          <td>0.064558</td>
          <td>0.039330</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.951126</td>
          <td>0.286461</td>
          <td>25.985304</td>
          <td>0.109482</td>
          <td>25.909274</td>
          <td>0.092562</td>
          <td>25.643675</td>
          <td>0.120160</td>
          <td>25.159004</td>
          <td>0.147509</td>
          <td>25.891243</td>
          <td>0.556905</td>
          <td>0.175365</td>
          <td>0.138129</td>
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
          <th>99995</th>
          <td>0.389450</td>
          <td>26.321766</td>
          <td>0.366308</td>
          <td>26.307285</td>
          <td>0.136025</td>
          <td>25.431493</td>
          <td>0.056607</td>
          <td>25.113268</td>
          <td>0.070237</td>
          <td>24.990940</td>
          <td>0.119190</td>
          <td>24.348278</td>
          <td>0.152209</td>
          <td>0.061324</td>
          <td>0.038267</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.743321</td>
          <td>0.195731</td>
          <td>25.785724</td>
          <td>0.076777</td>
          <td>25.270374</td>
          <td>0.079960</td>
          <td>24.800297</td>
          <td>0.100048</td>
          <td>24.047804</td>
          <td>0.116370</td>
          <td>0.003806</td>
          <td>0.002306</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.408377</td>
          <td>0.391739</td>
          <td>26.793541</td>
          <td>0.205681</td>
          <td>26.511506</td>
          <td>0.145924</td>
          <td>26.251808</td>
          <td>0.188755</td>
          <td>25.795306</td>
          <td>0.236142</td>
          <td>25.530400</td>
          <td>0.400674</td>
          <td>0.054374</td>
          <td>0.046719</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.276993</td>
          <td>0.397427</td>
          <td>26.131405</td>
          <td>0.135776</td>
          <td>26.010359</td>
          <td>0.111410</td>
          <td>25.947946</td>
          <td>0.172116</td>
          <td>26.015870</td>
          <td>0.329198</td>
          <td>25.431670</td>
          <td>0.430817</td>
          <td>0.266018</td>
          <td>0.231582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.823966</td>
          <td>0.532041</td>
          <td>26.820955</td>
          <td>0.208892</td>
          <td>26.404159</td>
          <td>0.131898</td>
          <td>26.785023</td>
          <td>0.290844</td>
          <td>26.521119</td>
          <td>0.417942</td>
          <td>25.588053</td>
          <td>0.415553</td>
          <td>0.002041</td>
          <td>0.001876</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.006840</td>
          <td>1.063880</td>
          <td>27.047895</td>
          <td>0.220823</td>
          <td>26.137245</td>
          <td>0.089196</td>
          <td>25.168306</td>
          <td>0.061794</td>
          <td>24.617673</td>
          <td>0.072622</td>
          <td>23.994234</td>
          <td>0.094266</td>
          <td>0.016082</td>
          <td>0.009486</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.362655</td>
          <td>1.331524</td>
          <td>28.145388</td>
          <td>0.545265</td>
          <td>26.716306</td>
          <td>0.156126</td>
          <td>26.584040</td>
          <td>0.223429</td>
          <td>26.337963</td>
          <td>0.331075</td>
          <td>24.975466</td>
          <td>0.231526</td>
          <td>0.078366</td>
          <td>0.056858</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.039888</td>
          <td>0.340329</td>
          <td>29.273037</td>
          <td>1.358084</td>
          <td>27.334021</td>
          <td>0.350119</td>
          <td>26.007555</td>
          <td>0.188390</td>
          <td>24.947721</td>
          <td>0.140733</td>
          <td>24.370321</td>
          <td>0.190279</td>
          <td>0.217427</td>
          <td>0.193605</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.728692</td>
          <td>3.352775</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.135567</td>
          <td>0.217779</td>
          <td>26.059500</td>
          <td>0.140021</td>
          <td>25.473227</td>
          <td>0.158529</td>
          <td>25.338872</td>
          <td>0.304786</td>
          <td>0.064558</td>
          <td>0.039330</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.803088</td>
          <td>0.256755</td>
          <td>26.075429</td>
          <td>0.119979</td>
          <td>25.879386</td>
          <td>0.091469</td>
          <td>25.773817</td>
          <td>0.136460</td>
          <td>25.307939</td>
          <td>0.169893</td>
          <td>25.290628</td>
          <td>0.358715</td>
          <td>0.175365</td>
          <td>0.138129</td>
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
          <th>99995</th>
          <td>0.389450</td>
          <td>26.720959</td>
          <td>0.455742</td>
          <td>26.344074</td>
          <td>0.124619</td>
          <td>25.464459</td>
          <td>0.050771</td>
          <td>24.981922</td>
          <td>0.054164</td>
          <td>24.926015</td>
          <td>0.098360</td>
          <td>24.797355</td>
          <td>0.194583</td>
          <td>0.061324</td>
          <td>0.038267</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.562489</td>
          <td>0.807428</td>
          <td>26.598462</td>
          <td>0.150741</td>
          <td>26.171205</td>
          <td>0.091695</td>
          <td>25.134817</td>
          <td>0.059844</td>
          <td>24.909517</td>
          <td>0.093719</td>
          <td>24.145350</td>
          <td>0.107358</td>
          <td>0.003806</td>
          <td>0.002306</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.585246</td>
          <td>0.411065</td>
          <td>26.488859</td>
          <td>0.141175</td>
          <td>26.389344</td>
          <td>0.114757</td>
          <td>26.659002</td>
          <td>0.231792</td>
          <td>25.889929</td>
          <td>0.224584</td>
          <td>25.994232</td>
          <td>0.503683</td>
          <td>0.054374</td>
          <td>0.046719</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.012152</td>
          <td>0.747390</td>
          <td>26.424496</td>
          <td>0.199448</td>
          <td>26.194293</td>
          <td>0.151163</td>
          <td>26.771597</td>
          <td>0.387803</td>
          <td>25.458279</td>
          <td>0.240237</td>
          <td>25.010828</td>
          <td>0.355314</td>
          <td>0.266018</td>
          <td>0.231582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.970570</td>
          <td>0.246816</td>
          <td>27.312666</td>
          <td>0.274050</td>
          <td>26.532235</td>
          <td>0.125677</td>
          <td>26.273315</td>
          <td>0.161964</td>
          <td>25.724024</td>
          <td>0.189288</td>
          <td>26.815669</td>
          <td>0.862181</td>
          <td>0.002041</td>
          <td>0.001876</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
