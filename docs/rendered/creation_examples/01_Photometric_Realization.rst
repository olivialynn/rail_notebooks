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

    <pzflow.flow.Flow at 0x7f79dc455f60>



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
    0      23.994413  0.091571  0.060010  
    1      25.391064  0.163856  0.101059  
    2      24.304707  0.122963  0.079726  
    3      25.291103  0.068563  0.036193  
    4      25.096743  0.035332  0.026936  
    ...          ...       ...       ...  
    99995  24.737946  0.086728  0.069765  
    99996  24.224169  0.039655  0.032837  
    99997  25.613836  0.033709  0.025411  
    99998  25.274899  0.095152  0.051962  
    99999  25.699642  0.089481  0.072217  
    
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
          <td>27.192385</td>
          <td>0.629074</td>
          <td>26.561473</td>
          <td>0.146014</td>
          <td>26.088837</td>
          <td>0.085272</td>
          <td>25.215613</td>
          <td>0.064280</td>
          <td>24.639261</td>
          <td>0.073846</td>
          <td>24.061601</td>
          <td>0.099759</td>
          <td>0.091571</td>
          <td>0.060010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.076456</td>
          <td>0.225690</td>
          <td>26.592188</td>
          <td>0.132368</td>
          <td>26.002363</td>
          <td>0.128279</td>
          <td>25.841797</td>
          <td>0.208969</td>
          <td>25.887332</td>
          <td>0.451546</td>
          <td>0.163856</td>
          <td>0.101059</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.196184</td>
          <td>0.487734</td>
          <td>26.180464</td>
          <td>0.149578</td>
          <td>24.984930</td>
          <td>0.100115</td>
          <td>24.295964</td>
          <td>0.122390</td>
          <td>0.122963</td>
          <td>0.079726</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.039998</td>
          <td>0.194042</td>
          <td>26.137090</td>
          <td>0.144104</td>
          <td>25.614799</td>
          <td>0.172554</td>
          <td>25.597538</td>
          <td>0.361408</td>
          <td>0.068563</td>
          <td>0.036193</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.344830</td>
          <td>0.333832</td>
          <td>25.945577</td>
          <td>0.085417</td>
          <td>25.909734</td>
          <td>0.072802</td>
          <td>25.829207</td>
          <td>0.110348</td>
          <td>25.416036</td>
          <td>0.145584</td>
          <td>24.792348</td>
          <td>0.187304</td>
          <td>0.035332</td>
          <td>0.026936</td>
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
          <td>26.974984</td>
          <td>0.538861</td>
          <td>26.266599</td>
          <td>0.113136</td>
          <td>25.412842</td>
          <td>0.046852</td>
          <td>25.104307</td>
          <td>0.058237</td>
          <td>24.889595</td>
          <td>0.092081</td>
          <td>24.877164</td>
          <td>0.201171</td>
          <td>0.086728</td>
          <td>0.069765</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.145771</td>
          <td>0.608839</td>
          <td>26.962575</td>
          <td>0.205243</td>
          <td>25.975718</td>
          <td>0.077174</td>
          <td>25.135048</td>
          <td>0.059847</td>
          <td>24.730709</td>
          <td>0.080058</td>
          <td>24.218701</td>
          <td>0.114436</td>
          <td>0.039655</td>
          <td>0.032837</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.599139</td>
          <td>0.826743</td>
          <td>26.846392</td>
          <td>0.186135</td>
          <td>26.525456</td>
          <td>0.124934</td>
          <td>26.574434</td>
          <td>0.208922</td>
          <td>25.921447</td>
          <td>0.223323</td>
          <td>25.355941</td>
          <td>0.298322</td>
          <td>0.033709</td>
          <td>0.025411</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.154827</td>
          <td>0.612732</td>
          <td>26.128119</td>
          <td>0.100254</td>
          <td>26.073151</td>
          <td>0.084102</td>
          <td>25.716189</td>
          <td>0.099964</td>
          <td>25.797561</td>
          <td>0.201364</td>
          <td>24.949479</td>
          <td>0.213727</td>
          <td>0.095152</td>
          <td>0.051962</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.618795</td>
          <td>0.413187</td>
          <td>26.932650</td>
          <td>0.200157</td>
          <td>26.628185</td>
          <td>0.136549</td>
          <td>26.411506</td>
          <td>0.182149</td>
          <td>26.165470</td>
          <td>0.272978</td>
          <td>25.704276</td>
          <td>0.392690</td>
          <td>0.089481</td>
          <td>0.072217</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.904143</td>
          <td>0.227728</td>
          <td>26.019464</td>
          <td>0.096219</td>
          <td>25.094041</td>
          <td>0.069860</td>
          <td>24.708170</td>
          <td>0.094138</td>
          <td>23.962384</td>
          <td>0.110240</td>
          <td>0.091571</td>
          <td>0.060010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.115169</td>
          <td>0.279814</td>
          <td>26.654383</td>
          <td>0.173098</td>
          <td>26.508791</td>
          <td>0.245613</td>
          <td>25.327446</td>
          <td>0.167214</td>
          <td>26.364203</td>
          <td>0.760552</td>
          <td>0.163856</td>
          <td>0.101059</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.500290</td>
          <td>0.861923</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.122367</td>
          <td>1.051801</td>
          <td>26.038018</td>
          <td>0.161629</td>
          <td>25.138379</td>
          <td>0.138966</td>
          <td>24.328717</td>
          <td>0.153668</td>
          <td>0.122963</td>
          <td>0.079726</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.347630</td>
          <td>0.769868</td>
          <td>28.502538</td>
          <td>0.756013</td>
          <td>27.660112</td>
          <td>0.376391</td>
          <td>26.240076</td>
          <td>0.187159</td>
          <td>25.424229</td>
          <td>0.173227</td>
          <td>25.422594</td>
          <td>0.369032</td>
          <td>0.068563</td>
          <td>0.036193</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.640177</td>
          <td>0.465573</td>
          <td>25.979073</td>
          <td>0.101788</td>
          <td>25.945933</td>
          <td>0.088716</td>
          <td>25.684771</td>
          <td>0.115393</td>
          <td>25.450590</td>
          <td>0.175996</td>
          <td>26.011318</td>
          <td>0.570325</td>
          <td>0.035332</td>
          <td>0.026936</td>
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
          <td>26.277539</td>
          <td>0.356720</td>
          <td>27.061378</td>
          <td>0.259371</td>
          <td>25.370389</td>
          <td>0.054267</td>
          <td>25.101255</td>
          <td>0.070359</td>
          <td>24.868364</td>
          <td>0.108384</td>
          <td>24.944769</td>
          <td>0.254161</td>
          <td>0.086728</td>
          <td>0.069765</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.599387</td>
          <td>0.902166</td>
          <td>26.527470</td>
          <td>0.163669</td>
          <td>26.209015</td>
          <td>0.111833</td>
          <td>25.098313</td>
          <td>0.069002</td>
          <td>24.905563</td>
          <td>0.110179</td>
          <td>24.245027</td>
          <td>0.138677</td>
          <td>0.039655</td>
          <td>0.032837</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.360561</td>
          <td>0.376070</td>
          <td>26.580194</td>
          <td>0.170961</td>
          <td>26.316903</td>
          <td>0.122659</td>
          <td>26.299140</td>
          <td>0.195337</td>
          <td>25.961035</td>
          <td>0.269131</td>
          <td>25.570687</td>
          <td>0.411173</td>
          <td>0.033709</td>
          <td>0.025411</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.904297</td>
          <td>0.570720</td>
          <td>26.394984</td>
          <td>0.148082</td>
          <td>26.116828</td>
          <td>0.104738</td>
          <td>26.046620</td>
          <td>0.160296</td>
          <td>25.875216</td>
          <td>0.254772</td>
          <td>25.104194</td>
          <td>0.289072</td>
          <td>0.095152</td>
          <td>0.051962</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.643997</td>
          <td>0.472592</td>
          <td>26.641842</td>
          <td>0.183138</td>
          <td>26.420389</td>
          <td>0.136684</td>
          <td>26.195506</td>
          <td>0.182385</td>
          <td>26.143635</td>
          <td>0.317309</td>
          <td>24.983938</td>
          <td>0.262783</td>
          <td>0.089481</td>
          <td>0.072217</td>
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

.. parsed-literal::

    




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
          <td>27.569171</td>
          <td>0.842906</td>
          <td>26.722985</td>
          <td>0.178557</td>
          <td>26.080248</td>
          <td>0.091187</td>
          <td>25.175530</td>
          <td>0.067114</td>
          <td>24.514300</td>
          <td>0.071267</td>
          <td>24.055017</td>
          <td>0.107109</td>
          <td>0.091571</td>
          <td>0.060010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.429240</td>
          <td>1.464534</td>
          <td>27.263061</td>
          <td>0.309182</td>
          <td>26.573970</td>
          <td>0.157959</td>
          <td>26.063578</td>
          <td>0.165149</td>
          <td>25.944574</td>
          <td>0.273525</td>
          <td>26.370638</td>
          <td>0.749867</td>
          <td>0.163856</td>
          <td>0.101059</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.000946</td>
          <td>2.738462</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.475724</td>
          <td>0.312146</td>
          <td>26.206028</td>
          <td>0.173581</td>
          <td>24.980688</td>
          <td>0.112967</td>
          <td>24.489899</td>
          <td>0.164200</td>
          <td>0.122963</td>
          <td>0.079726</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.171286</td>
          <td>0.547313</td>
          <td>27.077321</td>
          <td>0.207767</td>
          <td>26.157808</td>
          <td>0.152629</td>
          <td>25.755042</td>
          <td>0.201619</td>
          <td>25.604006</td>
          <td>0.376427</td>
          <td>0.068563</td>
          <td>0.036193</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.463308</td>
          <td>0.369410</td>
          <td>26.018703</td>
          <td>0.092139</td>
          <td>25.922686</td>
          <td>0.074628</td>
          <td>25.728439</td>
          <td>0.102451</td>
          <td>25.630489</td>
          <td>0.177115</td>
          <td>24.660548</td>
          <td>0.169736</td>
          <td>0.035332</td>
          <td>0.026936</td>
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
          <td>27.464696</td>
          <td>0.788948</td>
          <td>26.446031</td>
          <td>0.141285</td>
          <td>25.445117</td>
          <td>0.052144</td>
          <td>24.940333</td>
          <td>0.054647</td>
          <td>24.782369</td>
          <td>0.090538</td>
          <td>24.733681</td>
          <td>0.192513</td>
          <td>0.086728</td>
          <td>0.069765</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.890700</td>
          <td>1.702501</td>
          <td>26.742281</td>
          <td>0.172969</td>
          <td>25.820735</td>
          <td>0.068498</td>
          <td>25.149139</td>
          <td>0.061748</td>
          <td>24.600479</td>
          <td>0.072638</td>
          <td>24.256047</td>
          <td>0.120386</td>
          <td>0.039655</td>
          <td>0.032837</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.371973</td>
          <td>0.343620</td>
          <td>26.663891</td>
          <td>0.161021</td>
          <td>26.396528</td>
          <td>0.113013</td>
          <td>26.260994</td>
          <td>0.162226</td>
          <td>26.016788</td>
          <td>0.244391</td>
          <td>25.603308</td>
          <td>0.367074</td>
          <td>0.033709</td>
          <td>0.025411</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.517452</td>
          <td>0.399884</td>
          <td>26.268944</td>
          <td>0.120734</td>
          <td>26.034696</td>
          <td>0.087432</td>
          <td>25.934522</td>
          <td>0.130340</td>
          <td>25.451646</td>
          <td>0.161075</td>
          <td>25.803108</td>
          <td>0.451927</td>
          <td>0.095152</td>
          <td>0.051962</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.490002</td>
          <td>0.804143</td>
          <td>26.627315</td>
          <td>0.165694</td>
          <td>26.569942</td>
          <td>0.140759</td>
          <td>26.165082</td>
          <td>0.160508</td>
          <td>25.658734</td>
          <td>0.193804</td>
          <td>25.594765</td>
          <td>0.388830</td>
          <td>0.089481</td>
          <td>0.072217</td>
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
