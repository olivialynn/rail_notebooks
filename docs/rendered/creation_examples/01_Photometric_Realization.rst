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

    <pzflow.flow.Flow at 0x7faca2f45870>



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
    0      23.994413  0.087345  0.049395  
    1      25.391064  0.126808  0.107673  
    2      24.304707  0.028062  0.018295  
    3      25.291103  0.059101  0.058071  
    4      25.096743  0.019330  0.015243  
    ...          ...       ...       ...  
    99995  24.737946  0.024390  0.021835  
    99996  24.224169  0.106923  0.066487  
    99997  25.613836  0.070487  0.063776  
    99998  25.274899  0.077625  0.068587  
    99999  25.699642  0.088005  0.061631  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>26.632610</td>
          <td>0.417572</td>
          <td>26.801924</td>
          <td>0.179265</td>
          <td>26.048313</td>
          <td>0.082280</td>
          <td>25.256842</td>
          <td>0.066672</td>
          <td>24.652900</td>
          <td>0.074741</td>
          <td>23.922813</td>
          <td>0.088313</td>
          <td>0.087345</td>
          <td>0.049395</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.988019</td>
          <td>1.051013</td>
          <td>27.273955</td>
          <td>0.265534</td>
          <td>26.668338</td>
          <td>0.141360</td>
          <td>26.167180</td>
          <td>0.147881</td>
          <td>25.873786</td>
          <td>0.214630</td>
          <td>25.451842</td>
          <td>0.322127</td>
          <td>0.126808</td>
          <td>0.107673</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.469606</td>
          <td>0.759667</td>
          <td>28.661445</td>
          <td>0.749053</td>
          <td>27.827846</td>
          <td>0.368423</td>
          <td>26.026725</td>
          <td>0.131014</td>
          <td>25.277516</td>
          <td>0.129182</td>
          <td>24.367656</td>
          <td>0.130237</td>
          <td>0.028062</td>
          <td>0.018295</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.856667</td>
          <td>0.494143</td>
          <td>29.278614</td>
          <td>1.099837</td>
          <td>27.337369</td>
          <td>0.248554</td>
          <td>26.592847</td>
          <td>0.212164</td>
          <td>25.829154</td>
          <td>0.206769</td>
          <td>25.295394</td>
          <td>0.284093</td>
          <td>0.059101</td>
          <td>0.058071</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.559403</td>
          <td>0.394765</td>
          <td>26.027966</td>
          <td>0.091830</td>
          <td>25.943493</td>
          <td>0.075007</td>
          <td>25.628410</td>
          <td>0.092554</td>
          <td>25.300803</td>
          <td>0.131812</td>
          <td>25.587551</td>
          <td>0.358591</td>
          <td>0.019330</td>
          <td>0.015243</td>
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
          <td>26.955294</td>
          <td>0.531208</td>
          <td>26.320426</td>
          <td>0.118558</td>
          <td>25.432992</td>
          <td>0.047698</td>
          <td>25.063212</td>
          <td>0.056151</td>
          <td>24.845157</td>
          <td>0.088553</td>
          <td>24.804162</td>
          <td>0.189182</td>
          <td>0.024390</td>
          <td>0.021835</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.582427</td>
          <td>1.455810</td>
          <td>27.204392</td>
          <td>0.250836</td>
          <td>25.927681</td>
          <td>0.073966</td>
          <td>25.238509</td>
          <td>0.065598</td>
          <td>24.939803</td>
          <td>0.096231</td>
          <td>24.123841</td>
          <td>0.105344</td>
          <td>0.106923</td>
          <td>0.066487</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.318521</td>
          <td>0.326943</td>
          <td>26.936159</td>
          <td>0.200747</td>
          <td>26.263494</td>
          <td>0.099418</td>
          <td>26.159631</td>
          <td>0.146925</td>
          <td>25.626683</td>
          <td>0.174306</td>
          <td>25.937231</td>
          <td>0.468775</td>
          <td>0.070487</td>
          <td>0.063776</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.150917</td>
          <td>0.285875</td>
          <td>26.378010</td>
          <td>0.124633</td>
          <td>25.904414</td>
          <td>0.072460</td>
          <td>25.922754</td>
          <td>0.119716</td>
          <td>25.805871</td>
          <td>0.202773</td>
          <td>24.737921</td>
          <td>0.178874</td>
          <td>0.077625</td>
          <td>0.068587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.825132</td>
          <td>0.482733</td>
          <td>26.634537</td>
          <td>0.155453</td>
          <td>26.422867</td>
          <td>0.114274</td>
          <td>26.173869</td>
          <td>0.148733</td>
          <td>25.967500</td>
          <td>0.232023</td>
          <td>26.936548</td>
          <td>0.930037</td>
          <td>0.088005</td>
          <td>0.061631</td>
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
          <td>28.351601</td>
          <td>1.395229</td>
          <td>27.244079</td>
          <td>0.299827</td>
          <td>26.048235</td>
          <td>0.098366</td>
          <td>25.163299</td>
          <td>0.074027</td>
          <td>24.661281</td>
          <td>0.090055</td>
          <td>23.884478</td>
          <td>0.102655</td>
          <td>0.087345</td>
          <td>0.049395</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.449499</td>
          <td>1.485215</td>
          <td>27.547449</td>
          <td>0.389720</td>
          <td>26.524699</td>
          <td>0.152909</td>
          <td>26.481129</td>
          <td>0.236904</td>
          <td>25.971586</td>
          <td>0.282163</td>
          <td>25.285323</td>
          <td>0.342085</td>
          <td>0.126808</td>
          <td>0.107673</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.682221</td>
          <td>0.380150</td>
          <td>26.015151</td>
          <td>0.153293</td>
          <td>25.291279</td>
          <td>0.153424</td>
          <td>24.256431</td>
          <td>0.139682</td>
          <td>0.028062</td>
          <td>0.018295</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.155613</td>
          <td>2.025526</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.139945</td>
          <td>0.248448</td>
          <td>26.224789</td>
          <td>0.185026</td>
          <td>25.696372</td>
          <td>0.218119</td>
          <td>24.949960</td>
          <td>0.253033</td>
          <td>0.059101</td>
          <td>0.058071</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.991515</td>
          <td>0.280182</td>
          <td>26.075613</td>
          <td>0.110505</td>
          <td>25.938339</td>
          <td>0.087921</td>
          <td>25.459710</td>
          <td>0.094555</td>
          <td>25.584489</td>
          <td>0.196636</td>
          <td>25.104049</td>
          <td>0.284033</td>
          <td>0.019330</td>
          <td>0.015243</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.389059</td>
          <td>0.145044</td>
          <td>25.393343</td>
          <td>0.054338</td>
          <td>25.067840</td>
          <td>0.066979</td>
          <td>24.814408</td>
          <td>0.101473</td>
          <td>24.503355</td>
          <td>0.172543</td>
          <td>0.024390</td>
          <td>0.021835</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.052798</td>
          <td>0.299616</td>
          <td>26.642012</td>
          <td>0.183830</td>
          <td>26.121946</td>
          <td>0.105905</td>
          <td>25.064203</td>
          <td>0.068481</td>
          <td>24.857190</td>
          <td>0.107921</td>
          <td>24.330727</td>
          <td>0.152564</td>
          <td>0.106923</td>
          <td>0.066487</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.355817</td>
          <td>0.377764</td>
          <td>26.385088</td>
          <td>0.146264</td>
          <td>26.375286</td>
          <td>0.130578</td>
          <td>26.182714</td>
          <td>0.179194</td>
          <td>25.731183</td>
          <td>0.225293</td>
          <td>25.937491</td>
          <td>0.546203</td>
          <td>0.070487</td>
          <td>0.063776</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>29.471526</td>
          <td>2.303003</td>
          <td>26.119578</td>
          <td>0.116567</td>
          <td>26.327057</td>
          <td>0.125581</td>
          <td>25.609520</td>
          <td>0.109699</td>
          <td>25.747740</td>
          <td>0.229011</td>
          <td>25.386281</td>
          <td>0.361283</td>
          <td>0.077625</td>
          <td>0.068587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.327500</td>
          <td>0.763701</td>
          <td>26.592038</td>
          <td>0.175152</td>
          <td>26.880123</td>
          <td>0.201661</td>
          <td>26.491574</td>
          <td>0.233063</td>
          <td>27.094885</td>
          <td>0.645666</td>
          <td>25.044411</td>
          <td>0.275332</td>
          <td>0.088005</td>
          <td>0.061631</td>
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
          <td>28.385732</td>
          <td>1.350004</td>
          <td>26.892580</td>
          <td>0.204004</td>
          <td>25.985900</td>
          <td>0.082961</td>
          <td>25.145990</td>
          <td>0.064595</td>
          <td>24.667892</td>
          <td>0.080687</td>
          <td>23.866443</td>
          <td>0.089722</td>
          <td>0.087345</td>
          <td>0.049395</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.127050</td>
          <td>0.268200</td>
          <td>26.551554</td>
          <td>0.149256</td>
          <td>26.497808</td>
          <td>0.229146</td>
          <td>26.231640</td>
          <td>0.332585</td>
          <td>25.097967</td>
          <td>0.281446</td>
          <td>0.126808</td>
          <td>0.107673</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.544091</td>
          <td>1.939422</td>
          <td>25.923212</td>
          <td>0.120708</td>
          <td>25.230938</td>
          <td>0.124999</td>
          <td>24.444182</td>
          <td>0.140214</td>
          <td>0.028062</td>
          <td>0.018295</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.291425</td>
          <td>1.273233</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.026172</td>
          <td>0.200148</td>
          <td>26.524764</td>
          <td>0.209522</td>
          <td>25.484011</td>
          <td>0.161161</td>
          <td>25.103366</td>
          <td>0.253564</td>
          <td>0.059101</td>
          <td>0.058071</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.017088</td>
          <td>0.257082</td>
          <td>26.015247</td>
          <td>0.091134</td>
          <td>26.046922</td>
          <td>0.082519</td>
          <td>25.551192</td>
          <td>0.086850</td>
          <td>25.271058</td>
          <td>0.128981</td>
          <td>25.079875</td>
          <td>0.239129</td>
          <td>0.019330</td>
          <td>0.015243</td>
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
          <td>29.380426</td>
          <td>2.099638</td>
          <td>26.212447</td>
          <td>0.108595</td>
          <td>25.439550</td>
          <td>0.048329</td>
          <td>25.129456</td>
          <td>0.060010</td>
          <td>24.864119</td>
          <td>0.090696</td>
          <td>24.389327</td>
          <td>0.133686</td>
          <td>0.024390</td>
          <td>0.021835</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.068110</td>
          <td>0.242592</td>
          <td>25.999245</td>
          <td>0.086736</td>
          <td>25.171338</td>
          <td>0.068369</td>
          <td>24.905818</td>
          <td>0.102749</td>
          <td>24.481621</td>
          <td>0.158277</td>
          <td>0.106923</td>
          <td>0.066487</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.623053</td>
          <td>0.864697</td>
          <td>26.688770</td>
          <td>0.170988</td>
          <td>26.461331</td>
          <td>0.125111</td>
          <td>26.035753</td>
          <td>0.140129</td>
          <td>25.710763</td>
          <td>0.197817</td>
          <td>25.897833</td>
          <td>0.478927</td>
          <td>0.070487</td>
          <td>0.063776</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.499574</td>
          <td>0.393166</td>
          <td>26.127133</td>
          <td>0.106267</td>
          <td>26.059560</td>
          <td>0.088952</td>
          <td>26.065210</td>
          <td>0.145215</td>
          <td>26.218852</td>
          <td>0.303331</td>
          <td>26.377867</td>
          <td>0.680774</td>
          <td>0.077625</td>
          <td>0.068587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.842620</td>
          <td>0.510467</td>
          <td>26.801668</td>
          <td>0.190451</td>
          <td>26.572047</td>
          <td>0.139639</td>
          <td>26.184078</td>
          <td>0.161488</td>
          <td>25.468449</td>
          <td>0.163347</td>
          <td>25.198120</td>
          <td>0.281281</td>
          <td>0.088005</td>
          <td>0.061631</td>
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
