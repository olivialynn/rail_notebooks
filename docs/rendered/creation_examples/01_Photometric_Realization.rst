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

    <pzflow.flow.Flow at 0x7fe6540248b0>



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
    0      23.994413  0.003001  0.002448  
    1      25.391064  0.058013  0.046465  
    2      24.304707  0.020357  0.012453  
    3      25.291103  0.088002  0.060136  
    4      25.096743  0.106340  0.082473  
    ...          ...       ...       ...  
    99995  24.737946  0.034685  0.018277  
    99996  24.224169  0.087679  0.074763  
    99997  25.613836  0.128155  0.074532  
    99998  25.274899  0.081485  0.053148  
    99999  25.699642  0.035964  0.032161  
    
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
          <td>26.985545</td>
          <td>0.543001</td>
          <td>27.013359</td>
          <td>0.214144</td>
          <td>26.172967</td>
          <td>0.091824</td>
          <td>25.107385</td>
          <td>0.058396</td>
          <td>24.684531</td>
          <td>0.076860</td>
          <td>23.972378</td>
          <td>0.092247</td>
          <td>0.003001</td>
          <td>0.002448</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.475897</td>
          <td>1.378194</td>
          <td>27.259045</td>
          <td>0.262321</td>
          <td>26.465198</td>
          <td>0.118563</td>
          <td>26.592430</td>
          <td>0.212090</td>
          <td>25.654693</td>
          <td>0.178498</td>
          <td>25.054102</td>
          <td>0.233150</td>
          <td>0.058013</td>
          <td>0.046465</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.134812</td>
          <td>1.888488</td>
          <td>29.040925</td>
          <td>0.954616</td>
          <td>28.684877</td>
          <td>0.690775</td>
          <td>25.953789</td>
          <td>0.122988</td>
          <td>25.030970</td>
          <td>0.104232</td>
          <td>24.126700</td>
          <td>0.105607</td>
          <td>0.020357</td>
          <td>0.012453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.263563</td>
          <td>0.233871</td>
          <td>26.126978</td>
          <td>0.142855</td>
          <td>25.479577</td>
          <td>0.153746</td>
          <td>25.021594</td>
          <td>0.226951</td>
          <td>0.088002</td>
          <td>0.060136</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.204056</td>
          <td>0.298380</td>
          <td>26.147900</td>
          <td>0.102004</td>
          <td>25.956206</td>
          <td>0.075855</td>
          <td>25.760260</td>
          <td>0.103897</td>
          <td>25.677287</td>
          <td>0.181948</td>
          <td>25.482819</td>
          <td>0.330158</td>
          <td>0.106340</td>
          <td>0.082473</td>
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
          <td>27.126383</td>
          <td>0.600567</td>
          <td>26.297443</td>
          <td>0.116213</td>
          <td>25.439084</td>
          <td>0.047957</td>
          <td>24.938339</td>
          <td>0.050258</td>
          <td>24.814388</td>
          <td>0.086187</td>
          <td>24.404896</td>
          <td>0.134499</td>
          <td>0.034685</td>
          <td>0.018277</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.364181</td>
          <td>0.707926</td>
          <td>26.526432</td>
          <td>0.141680</td>
          <td>25.978028</td>
          <td>0.077332</td>
          <td>25.149591</td>
          <td>0.060625</td>
          <td>24.810526</td>
          <td>0.085894</td>
          <td>24.134987</td>
          <td>0.106375</td>
          <td>0.087679</td>
          <td>0.074763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.184805</td>
          <td>0.293796</td>
          <td>26.761300</td>
          <td>0.173194</td>
          <td>26.281735</td>
          <td>0.101019</td>
          <td>26.276280</td>
          <td>0.162366</td>
          <td>25.736579</td>
          <td>0.191294</td>
          <td>26.831734</td>
          <td>0.870978</td>
          <td>0.128155</td>
          <td>0.074532</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.395030</td>
          <td>0.347321</td>
          <td>26.086143</td>
          <td>0.096636</td>
          <td>26.084924</td>
          <td>0.084979</td>
          <td>26.008578</td>
          <td>0.128972</td>
          <td>25.817656</td>
          <td>0.204787</td>
          <td>25.245696</td>
          <td>0.272861</td>
          <td>0.081485</td>
          <td>0.053148</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.785936</td>
          <td>0.930230</td>
          <td>26.710839</td>
          <td>0.165918</td>
          <td>26.664995</td>
          <td>0.140954</td>
          <td>26.553878</td>
          <td>0.205357</td>
          <td>25.881875</td>
          <td>0.216084</td>
          <td>25.903784</td>
          <td>0.457169</td>
          <td>0.035964</td>
          <td>0.032161</td>
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
          <td>26.776208</td>
          <td>0.513815</td>
          <td>27.043785</td>
          <td>0.251257</td>
          <td>26.084002</td>
          <td>0.099814</td>
          <td>25.177068</td>
          <td>0.073635</td>
          <td>24.778591</td>
          <td>0.098163</td>
          <td>24.198812</td>
          <td>0.132653</td>
          <td>0.003001</td>
          <td>0.002448</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.455251</td>
          <td>2.282116</td>
          <td>27.252925</td>
          <td>0.300107</td>
          <td>26.457604</td>
          <td>0.139396</td>
          <td>26.087059</td>
          <td>0.164235</td>
          <td>25.064871</td>
          <td>0.127151</td>
          <td>25.907561</td>
          <td>0.531830</td>
          <td>0.058013</td>
          <td>0.046465</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.025114</td>
          <td>2.621413</td>
          <td>28.849674</td>
          <td>0.869315</td>
          <td>26.129272</td>
          <td>0.168828</td>
          <td>25.011584</td>
          <td>0.120410</td>
          <td>24.078452</td>
          <td>0.119627</td>
          <td>0.020357</td>
          <td>0.012453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.452995</td>
          <td>2.130498</td>
          <td>27.040023</td>
          <td>0.230367</td>
          <td>26.100040</td>
          <td>0.167672</td>
          <td>25.884120</td>
          <td>0.256500</td>
          <td>25.110801</td>
          <td>0.290460</td>
          <td>0.088002</td>
          <td>0.060136</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.051181</td>
          <td>0.637118</td>
          <td>26.077589</td>
          <td>0.113597</td>
          <td>25.947516</td>
          <td>0.091231</td>
          <td>25.569640</td>
          <td>0.107252</td>
          <td>25.257815</td>
          <td>0.153206</td>
          <td>24.633340</td>
          <td>0.197990</td>
          <td>0.106340</td>
          <td>0.082473</td>
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
          <td>26.507780</td>
          <td>0.421074</td>
          <td>26.241506</td>
          <td>0.127806</td>
          <td>25.499975</td>
          <td>0.059776</td>
          <td>25.022165</td>
          <td>0.064376</td>
          <td>24.880337</td>
          <td>0.107578</td>
          <td>24.538295</td>
          <td>0.177876</td>
          <td>0.034685</td>
          <td>0.018277</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.755182</td>
          <td>1.704894</td>
          <td>26.961221</td>
          <td>0.239177</td>
          <td>26.007989</td>
          <td>0.095461</td>
          <td>25.086330</td>
          <td>0.069541</td>
          <td>24.939447</td>
          <td>0.115479</td>
          <td>24.278725</td>
          <td>0.145312</td>
          <td>0.087679</td>
          <td>0.074763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.341506</td>
          <td>0.778187</td>
          <td>26.738495</td>
          <td>0.201062</td>
          <td>26.937115</td>
          <td>0.214866</td>
          <td>26.190660</td>
          <td>0.184141</td>
          <td>25.641369</td>
          <td>0.213176</td>
          <td>24.937610</td>
          <td>0.256354</td>
          <td>0.128155</td>
          <td>0.074532</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.724756</td>
          <td>0.499757</td>
          <td>26.575770</td>
          <td>0.172236</td>
          <td>26.090555</td>
          <td>0.101983</td>
          <td>25.988384</td>
          <td>0.151936</td>
          <td>26.288775</td>
          <td>0.353976</td>
          <td>25.370187</td>
          <td>0.356063</td>
          <td>0.081485</td>
          <td>0.053148</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.710570</td>
          <td>0.490794</td>
          <td>26.709957</td>
          <td>0.190957</td>
          <td>26.740456</td>
          <td>0.176628</td>
          <td>26.483075</td>
          <td>0.228002</td>
          <td>25.822806</td>
          <td>0.240497</td>
          <td>26.218156</td>
          <td>0.659938</td>
          <td>0.035964</td>
          <td>0.032161</td>
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
          <td>28.262135</td>
          <td>1.228968</td>
          <td>26.685315</td>
          <td>0.162360</td>
          <td>26.016370</td>
          <td>0.080002</td>
          <td>25.217994</td>
          <td>0.064423</td>
          <td>24.693739</td>
          <td>0.077495</td>
          <td>24.021334</td>
          <td>0.096309</td>
          <td>0.003001</td>
          <td>0.002448</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.321554</td>
          <td>0.284148</td>
          <td>26.322940</td>
          <td>0.108559</td>
          <td>26.611278</td>
          <td>0.223320</td>
          <td>25.981153</td>
          <td>0.242745</td>
          <td>25.661039</td>
          <td>0.392498</td>
          <td>0.058013</td>
          <td>0.046465</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.653663</td>
          <td>0.678230</td>
          <td>25.727822</td>
          <td>0.101397</td>
          <td>25.037501</td>
          <td>0.105232</td>
          <td>24.484878</td>
          <td>0.144668</td>
          <td>0.020357</td>
          <td>0.012453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.125494</td>
          <td>0.541976</td>
          <td>27.834853</td>
          <td>0.394384</td>
          <td>26.172303</td>
          <td>0.159689</td>
          <td>25.856168</td>
          <td>0.226196</td>
          <td>25.515627</td>
          <td>0.361903</td>
          <td>0.088002</td>
          <td>0.060136</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.147532</td>
          <td>0.306225</td>
          <td>26.069836</td>
          <td>0.104822</td>
          <td>25.939880</td>
          <td>0.083426</td>
          <td>25.750854</td>
          <td>0.115386</td>
          <td>25.373011</td>
          <td>0.156037</td>
          <td>25.104252</td>
          <td>0.269944</td>
          <td>0.106340</td>
          <td>0.082473</td>
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
          <td>26.729483</td>
          <td>0.452168</td>
          <td>26.366713</td>
          <td>0.124514</td>
          <td>25.448654</td>
          <td>0.048874</td>
          <td>24.948788</td>
          <td>0.051285</td>
          <td>24.772710</td>
          <td>0.083942</td>
          <td>24.683589</td>
          <td>0.172587</td>
          <td>0.034685</td>
          <td>0.018277</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.090109</td>
          <td>1.158654</td>
          <td>26.511520</td>
          <td>0.150135</td>
          <td>26.059073</td>
          <td>0.090234</td>
          <td>25.239788</td>
          <td>0.071652</td>
          <td>24.796530</td>
          <td>0.092150</td>
          <td>24.286359</td>
          <td>0.132059</td>
          <td>0.087679</td>
          <td>0.074763</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.077073</td>
          <td>0.624263</td>
          <td>26.488177</td>
          <td>0.152959</td>
          <td>26.379266</td>
          <td>0.124720</td>
          <td>25.964352</td>
          <td>0.141357</td>
          <td>25.927813</td>
          <td>0.252853</td>
          <td>25.741069</td>
          <td>0.452445</td>
          <td>0.128155</td>
          <td>0.074532</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.939371</td>
          <td>0.250270</td>
          <td>26.210246</td>
          <td>0.113459</td>
          <td>25.937051</td>
          <td>0.079210</td>
          <td>25.985419</td>
          <td>0.134453</td>
          <td>25.559532</td>
          <td>0.174431</td>
          <td>25.146321</td>
          <td>0.266526</td>
          <td>0.081485</td>
          <td>0.053148</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.520336</td>
          <td>0.386694</td>
          <td>26.936208</td>
          <td>0.203363</td>
          <td>26.917040</td>
          <td>0.177526</td>
          <td>26.718985</td>
          <td>0.239267</td>
          <td>26.405677</td>
          <td>0.335778</td>
          <td>28.804978</td>
          <td>2.349151</td>
          <td>0.035964</td>
          <td>0.032161</td>
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
