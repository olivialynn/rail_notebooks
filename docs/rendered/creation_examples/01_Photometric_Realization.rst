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

    <pzflow.flow.Flow at 0x7f8279b67a00>



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
    0      23.994413  0.058683  0.032030  
    1      25.391064  0.131911  0.103794  
    2      24.304707  0.020760  0.010705  
    3      25.291103  0.114874  0.082256  
    4      25.096743  0.109104  0.076550  
    ...          ...       ...       ...  
    99995  24.737946  0.122046  0.080802  
    99996  24.224169  0.099752  0.056957  
    99997  25.613836  0.023080  0.014153  
    99998  25.274899  0.042694  0.032139  
    99999  25.699642  0.090402  0.047266  
    
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
          <td>26.395149</td>
          <td>0.347353</td>
          <td>26.726963</td>
          <td>0.168211</td>
          <td>25.965549</td>
          <td>0.076484</td>
          <td>25.143491</td>
          <td>0.060297</td>
          <td>24.731009</td>
          <td>0.080079</td>
          <td>24.064083</td>
          <td>0.099976</td>
          <td>0.058683</td>
          <td>0.032030</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.676243</td>
          <td>0.431674</td>
          <td>27.397906</td>
          <td>0.293616</td>
          <td>26.624997</td>
          <td>0.136174</td>
          <td>26.338863</td>
          <td>0.171260</td>
          <td>25.823969</td>
          <td>0.205873</td>
          <td>25.461833</td>
          <td>0.324699</td>
          <td>0.131911</td>
          <td>0.103794</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.847446</td>
          <td>1.657453</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.453322</td>
          <td>0.587920</td>
          <td>26.012349</td>
          <td>0.129393</td>
          <td>25.034622</td>
          <td>0.104565</td>
          <td>24.349874</td>
          <td>0.128248</td>
          <td>0.020760</td>
          <td>0.010705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.435678</td>
          <td>0.642430</td>
          <td>27.501604</td>
          <td>0.284204</td>
          <td>26.418066</td>
          <td>0.183163</td>
          <td>25.393328</td>
          <td>0.142767</td>
          <td>25.362914</td>
          <td>0.300000</td>
          <td>0.114874</td>
          <td>0.082256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.662871</td>
          <td>0.427311</td>
          <td>25.960073</td>
          <td>0.086513</td>
          <td>25.972851</td>
          <td>0.076979</td>
          <td>25.652862</td>
          <td>0.094563</td>
          <td>25.330489</td>
          <td>0.135237</td>
          <td>25.433828</td>
          <td>0.317534</td>
          <td>0.109104</td>
          <td>0.076550</td>
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
          <td>26.957784</td>
          <td>0.532171</td>
          <td>26.330600</td>
          <td>0.119611</td>
          <td>25.408788</td>
          <td>0.046684</td>
          <td>25.079834</td>
          <td>0.056986</td>
          <td>24.781032</td>
          <td>0.083691</td>
          <td>24.566092</td>
          <td>0.154507</td>
          <td>0.122046</td>
          <td>0.080802</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.767240</td>
          <td>0.919516</td>
          <td>26.734258</td>
          <td>0.169258</td>
          <td>26.178072</td>
          <td>0.092237</td>
          <td>25.187818</td>
          <td>0.062715</td>
          <td>24.830373</td>
          <td>0.087408</td>
          <td>24.099414</td>
          <td>0.103117</td>
          <td>0.099752</td>
          <td>0.056957</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.702061</td>
          <td>0.440200</td>
          <td>26.615817</td>
          <td>0.152982</td>
          <td>26.478417</td>
          <td>0.119934</td>
          <td>26.052144</td>
          <td>0.133925</td>
          <td>25.747744</td>
          <td>0.193103</td>
          <td>25.011488</td>
          <td>0.225055</td>
          <td>0.023080</td>
          <td>0.014153</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.692760</td>
          <td>0.877618</td>
          <td>26.176236</td>
          <td>0.104562</td>
          <td>26.150035</td>
          <td>0.089992</td>
          <td>25.828911</td>
          <td>0.110320</td>
          <td>25.803306</td>
          <td>0.202337</td>
          <td>25.682390</td>
          <td>0.386097</td>
          <td>0.042694</td>
          <td>0.032139</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.292462</td>
          <td>0.320241</td>
          <td>26.958067</td>
          <td>0.204469</td>
          <td>26.611820</td>
          <td>0.134633</td>
          <td>26.275495</td>
          <td>0.162257</td>
          <td>25.670848</td>
          <td>0.180958</td>
          <td>25.864303</td>
          <td>0.443769</td>
          <td>0.090402</td>
          <td>0.047266</td>
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
          <td>28.104698</td>
          <td>1.217061</td>
          <td>27.001038</td>
          <td>0.244124</td>
          <td>26.103301</td>
          <td>0.102278</td>
          <td>25.169191</td>
          <td>0.073700</td>
          <td>24.932296</td>
          <td>0.113118</td>
          <td>24.208292</td>
          <td>0.134764</td>
          <td>0.058683</td>
          <td>0.032030</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.740720</td>
          <td>1.005709</td>
          <td>28.577116</td>
          <td>0.813197</td>
          <td>26.469279</td>
          <td>0.145908</td>
          <td>26.218873</td>
          <td>0.190442</td>
          <td>25.624046</td>
          <td>0.212097</td>
          <td>25.845202</td>
          <td>0.524147</td>
          <td>0.131911</td>
          <td>0.103794</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.580572</td>
          <td>0.729354</td>
          <td>25.949494</td>
          <td>0.144748</td>
          <td>25.213080</td>
          <td>0.143323</td>
          <td>24.216612</td>
          <td>0.134832</td>
          <td>0.020760</td>
          <td>0.010705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.634309</td>
          <td>0.936163</td>
          <td>27.716932</td>
          <td>0.439428</td>
          <td>27.154593</td>
          <td>0.256482</td>
          <td>26.210151</td>
          <td>0.186643</td>
          <td>25.395544</td>
          <td>0.172798</td>
          <td>25.642848</td>
          <td>0.445996</td>
          <td>0.114874</td>
          <td>0.082256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.169429</td>
          <td>0.329545</td>
          <td>26.033532</td>
          <td>0.109263</td>
          <td>25.981486</td>
          <td>0.093937</td>
          <td>25.738624</td>
          <td>0.124171</td>
          <td>25.491055</td>
          <td>0.186730</td>
          <td>24.881343</td>
          <td>0.243251</td>
          <td>0.109104</td>
          <td>0.076550</td>
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
          <td>29.571637</td>
          <td>2.404886</td>
          <td>26.263905</td>
          <td>0.134122</td>
          <td>25.425609</td>
          <td>0.057826</td>
          <td>25.092028</td>
          <td>0.070839</td>
          <td>24.817458</td>
          <td>0.105162</td>
          <td>24.538727</td>
          <td>0.183738</td>
          <td>0.122046</td>
          <td>0.080802</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.977025</td>
          <td>0.242265</td>
          <td>26.168621</td>
          <td>0.109840</td>
          <td>25.126738</td>
          <td>0.072053</td>
          <td>25.023745</td>
          <td>0.124224</td>
          <td>24.082580</td>
          <td>0.122631</td>
          <td>0.099752</td>
          <td>0.056957</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.417740</td>
          <td>1.433560</td>
          <td>26.629924</td>
          <td>0.178053</td>
          <td>26.240732</td>
          <td>0.114596</td>
          <td>26.506266</td>
          <td>0.231814</td>
          <td>26.077152</td>
          <td>0.295199</td>
          <td>25.778219</td>
          <td>0.480192</td>
          <td>0.023080</td>
          <td>0.014153</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.460582</td>
          <td>0.826228</td>
          <td>26.196836</td>
          <td>0.123200</td>
          <td>26.054558</td>
          <td>0.097740</td>
          <td>25.802960</td>
          <td>0.128060</td>
          <td>25.770090</td>
          <td>0.230431</td>
          <td>24.877947</td>
          <td>0.236926</td>
          <td>0.042694</td>
          <td>0.032139</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.686349</td>
          <td>0.486238</td>
          <td>26.780533</td>
          <td>0.204975</td>
          <td>26.221107</td>
          <td>0.114465</td>
          <td>26.057794</td>
          <td>0.161476</td>
          <td>25.537521</td>
          <td>0.191983</td>
          <td>24.988554</td>
          <td>0.262599</td>
          <td>0.090402</td>
          <td>0.047266</td>
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
          <td>26.832822</td>
          <td>0.493932</td>
          <td>26.923344</td>
          <td>0.203484</td>
          <td>26.039049</td>
          <td>0.084041</td>
          <td>25.176825</td>
          <td>0.064060</td>
          <td>24.838271</td>
          <td>0.090629</td>
          <td>24.091933</td>
          <td>0.105586</td>
          <td>0.058683</td>
          <td>0.032030</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.411897</td>
          <td>0.337667</td>
          <td>26.431515</td>
          <td>0.134843</td>
          <td>26.229148</td>
          <td>0.183289</td>
          <td>25.610431</td>
          <td>0.200472</td>
          <td>25.338991</td>
          <td>0.341904</td>
          <td>0.131911</td>
          <td>0.103794</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.306835</td>
          <td>0.682209</td>
          <td>31.442847</td>
          <td>2.858639</td>
          <td>32.398781</td>
          <td>3.629586</td>
          <td>26.229818</td>
          <td>0.156633</td>
          <td>25.092732</td>
          <td>0.110414</td>
          <td>24.384163</td>
          <td>0.132610</td>
          <td>0.020760</td>
          <td>0.010705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.640664</td>
          <td>0.902561</td>
          <td>27.381198</td>
          <td>0.318585</td>
          <td>28.176877</td>
          <td>0.530800</td>
          <td>26.129019</td>
          <td>0.161428</td>
          <td>25.181118</td>
          <td>0.133507</td>
          <td>25.482004</td>
          <td>0.368131</td>
          <td>0.114874</td>
          <td>0.082256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.064224</td>
          <td>0.285946</td>
          <td>26.301678</td>
          <td>0.127967</td>
          <td>26.035914</td>
          <td>0.090567</td>
          <td>25.643174</td>
          <td>0.104778</td>
          <td>25.397922</td>
          <td>0.159029</td>
          <td>24.841094</td>
          <td>0.216792</td>
          <td>0.109104</td>
          <td>0.076550</td>
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
          <td>27.275587</td>
          <td>0.714712</td>
          <td>26.464179</td>
          <td>0.149626</td>
          <td>25.487936</td>
          <td>0.056851</td>
          <td>25.051713</td>
          <td>0.063431</td>
          <td>24.958319</td>
          <td>0.110777</td>
          <td>25.000507</td>
          <td>0.251860</td>
          <td>0.122046</td>
          <td>0.080802</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.687848</td>
          <td>0.911557</td>
          <td>26.684495</td>
          <td>0.173757</td>
          <td>25.855143</td>
          <td>0.075245</td>
          <td>25.170869</td>
          <td>0.067275</td>
          <td>24.862957</td>
          <td>0.097500</td>
          <td>24.264952</td>
          <td>0.129383</td>
          <td>0.099752</td>
          <td>0.056957</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.572679</td>
          <td>0.400023</td>
          <td>26.650260</td>
          <td>0.158218</td>
          <td>26.367627</td>
          <td>0.109436</td>
          <td>26.102268</td>
          <td>0.140562</td>
          <td>25.903569</td>
          <td>0.221059</td>
          <td>25.400112</td>
          <td>0.310542</td>
          <td>0.023080</td>
          <td>0.014153</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.363092</td>
          <td>0.342736</td>
          <td>26.470371</td>
          <td>0.137206</td>
          <td>25.985044</td>
          <td>0.079314</td>
          <td>25.686821</td>
          <td>0.099383</td>
          <td>25.716724</td>
          <td>0.191575</td>
          <td>25.216432</td>
          <td>0.271352</td>
          <td>0.042694</td>
          <td>0.032139</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.586553</td>
          <td>0.848037</td>
          <td>26.799199</td>
          <td>0.188878</td>
          <td>26.429904</td>
          <td>0.122580</td>
          <td>26.232464</td>
          <td>0.167026</td>
          <td>26.354096</td>
          <td>0.336900</td>
          <td>25.849346</td>
          <td>0.464696</td>
          <td>0.090402</td>
          <td>0.047266</td>
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
