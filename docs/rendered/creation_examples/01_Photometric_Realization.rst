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

    <pzflow.flow.Flow at 0x7fd514bc84f0>



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
    0      23.994413  0.132353  0.080723  
    1      25.391064  0.097094  0.053152  
    2      24.304707  0.119666  0.085456  
    3      25.291103  0.109217  0.096802  
    4      25.096743  0.022166  0.016141  
    ...          ...       ...       ...  
    99995  24.737946  0.128220  0.108351  
    99996  24.224169  0.083154  0.072578  
    99997  25.613836  0.225915  0.203744  
    99998  25.274899  0.022843  0.021318  
    99999  25.699642  0.067755  0.053839  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.760265</td>
          <td>0.173041</td>
          <td>26.064788</td>
          <td>0.083484</td>
          <td>25.261523</td>
          <td>0.066949</td>
          <td>24.731919</td>
          <td>0.080144</td>
          <td>23.862060</td>
          <td>0.083713</td>
          <td>0.132353</td>
          <td>0.080723</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.800741</td>
          <td>0.938771</td>
          <td>28.299996</td>
          <td>0.583942</td>
          <td>26.859490</td>
          <td>0.166522</td>
          <td>25.986660</td>
          <td>0.126545</td>
          <td>26.107681</td>
          <td>0.260406</td>
          <td>25.249951</td>
          <td>0.273807</td>
          <td>0.097094</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.050241</td>
          <td>1.819265</td>
          <td>28.177395</td>
          <td>0.534636</td>
          <td>28.078449</td>
          <td>0.446605</td>
          <td>26.068575</td>
          <td>0.135840</td>
          <td>24.875675</td>
          <td>0.090962</td>
          <td>24.096643</td>
          <td>0.102867</td>
          <td>0.119666</td>
          <td>0.085456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.972178</td>
          <td>0.537765</td>
          <td>28.116386</td>
          <td>0.511330</td>
          <td>27.657516</td>
          <td>0.322115</td>
          <td>26.116207</td>
          <td>0.141536</td>
          <td>25.440574</td>
          <td>0.148687</td>
          <td>25.127438</td>
          <td>0.247697</td>
          <td>0.109217</td>
          <td>0.096802</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.977364</td>
          <td>0.248189</td>
          <td>25.936493</td>
          <td>0.084737</td>
          <td>25.842265</td>
          <td>0.068582</td>
          <td>25.738053</td>
          <td>0.101897</td>
          <td>25.353758</td>
          <td>0.137981</td>
          <td>24.963143</td>
          <td>0.216177</td>
          <td>0.022166</td>
          <td>0.016141</td>
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
          <td>26.778616</td>
          <td>0.466285</td>
          <td>26.592806</td>
          <td>0.149994</td>
          <td>25.423825</td>
          <td>0.047311</td>
          <td>25.045298</td>
          <td>0.055265</td>
          <td>24.765567</td>
          <td>0.082558</td>
          <td>24.763788</td>
          <td>0.182836</td>
          <td>0.128220</td>
          <td>0.108351</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.692235</td>
          <td>0.877327</td>
          <td>26.906983</td>
          <td>0.195887</td>
          <td>26.161666</td>
          <td>0.090917</td>
          <td>25.128534</td>
          <td>0.059502</td>
          <td>24.814290</td>
          <td>0.086179</td>
          <td>24.262308</td>
          <td>0.118862</td>
          <td>0.083154</td>
          <td>0.072578</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.703359</td>
          <td>0.883503</td>
          <td>26.519301</td>
          <td>0.140813</td>
          <td>26.232237</td>
          <td>0.096730</td>
          <td>26.415934</td>
          <td>0.182833</td>
          <td>26.042937</td>
          <td>0.246934</td>
          <td>25.426420</td>
          <td>0.315662</td>
          <td>0.225915</td>
          <td>0.203744</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.077095</td>
          <td>0.269267</td>
          <td>26.100639</td>
          <td>0.097871</td>
          <td>25.994538</td>
          <td>0.078467</td>
          <td>25.654292</td>
          <td>0.094682</td>
          <td>25.530734</td>
          <td>0.160625</td>
          <td>25.106951</td>
          <td>0.243554</td>
          <td>0.022843</td>
          <td>0.021318</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.246526</td>
          <td>0.653196</td>
          <td>27.214438</td>
          <td>0.252913</td>
          <td>26.822859</td>
          <td>0.161398</td>
          <td>26.249348</td>
          <td>0.158672</td>
          <td>25.810991</td>
          <td>0.203646</td>
          <td>25.367905</td>
          <td>0.301206</td>
          <td>0.067755</td>
          <td>0.053839</td>
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
          <td>27.141223</td>
          <td>0.681726</td>
          <td>27.002743</td>
          <td>0.251074</td>
          <td>26.021927</td>
          <td>0.098280</td>
          <td>25.129091</td>
          <td>0.073503</td>
          <td>24.714528</td>
          <td>0.096483</td>
          <td>23.974975</td>
          <td>0.113640</td>
          <td>0.132353</td>
          <td>0.080723</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.948147</td>
          <td>1.121248</td>
          <td>27.466512</td>
          <td>0.358764</td>
          <td>26.703588</td>
          <td>0.173915</td>
          <td>26.388759</td>
          <td>0.214198</td>
          <td>26.059433</td>
          <td>0.296143</td>
          <td>24.844465</td>
          <td>0.233929</td>
          <td>0.097094</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.738160</td>
          <td>0.511102</td>
          <td>28.343431</td>
          <td>0.691325</td>
          <td>27.321627</td>
          <td>0.294487</td>
          <td>25.789551</td>
          <td>0.130583</td>
          <td>25.118326</td>
          <td>0.136624</td>
          <td>24.604740</td>
          <td>0.194333</td>
          <td>0.119666</td>
          <td>0.085456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.573769</td>
          <td>0.452211</td>
          <td>27.114227</td>
          <td>0.274075</td>
          <td>27.313036</td>
          <td>0.292347</td>
          <td>26.428575</td>
          <td>0.224620</td>
          <td>25.564779</td>
          <td>0.199783</td>
          <td>25.525588</td>
          <td>0.408725</td>
          <td>0.109217</td>
          <td>0.096802</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.544905</td>
          <td>0.432750</td>
          <td>26.098952</td>
          <td>0.112799</td>
          <td>25.927866</td>
          <td>0.087136</td>
          <td>25.761928</td>
          <td>0.123136</td>
          <td>25.752500</td>
          <td>0.226332</td>
          <td>24.913472</td>
          <td>0.243137</td>
          <td>0.022166</td>
          <td>0.016141</td>
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
          <td>27.993604</td>
          <td>1.165319</td>
          <td>26.379617</td>
          <td>0.149577</td>
          <td>25.435329</td>
          <td>0.058963</td>
          <td>25.036160</td>
          <td>0.068180</td>
          <td>24.938076</td>
          <td>0.118070</td>
          <td>24.789323</td>
          <td>0.229042</td>
          <td>0.128220</td>
          <td>0.108351</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.824763</td>
          <td>0.539208</td>
          <td>26.673384</td>
          <td>0.187806</td>
          <td>26.187235</td>
          <td>0.111467</td>
          <td>25.153435</td>
          <td>0.073656</td>
          <td>24.755777</td>
          <td>0.098188</td>
          <td>24.213228</td>
          <td>0.137097</td>
          <td>0.083154</td>
          <td>0.072578</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.977674</td>
          <td>1.969514</td>
          <td>26.808516</td>
          <td>0.232570</td>
          <td>26.100977</td>
          <td>0.115995</td>
          <td>26.382709</td>
          <td>0.238556</td>
          <td>26.074433</td>
          <td>0.332968</td>
          <td>25.045270</td>
          <td>0.307528</td>
          <td>0.225915</td>
          <td>0.203744</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.075861</td>
          <td>0.300028</td>
          <td>26.032668</td>
          <td>0.106504</td>
          <td>26.064024</td>
          <td>0.098242</td>
          <td>25.718998</td>
          <td>0.118675</td>
          <td>25.628758</td>
          <td>0.204205</td>
          <td>24.976336</td>
          <td>0.256123</td>
          <td>0.022843</td>
          <td>0.021318</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.089450</td>
          <td>0.647464</td>
          <td>27.249817</td>
          <td>0.300168</td>
          <td>26.395682</td>
          <td>0.132566</td>
          <td>26.470956</td>
          <td>0.227634</td>
          <td>26.035986</td>
          <td>0.288542</td>
          <td>25.423053</td>
          <td>0.370003</td>
          <td>0.067755</td>
          <td>0.053839</td>
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
          <td>27.213002</td>
          <td>0.689960</td>
          <td>26.469777</td>
          <td>0.151965</td>
          <td>26.065031</td>
          <td>0.095813</td>
          <td>25.303738</td>
          <td>0.080278</td>
          <td>24.704441</td>
          <td>0.089770</td>
          <td>24.046286</td>
          <td>0.113358</td>
          <td>0.132353</td>
          <td>0.080723</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.592632</td>
          <td>0.363794</td>
          <td>26.670951</td>
          <td>0.152500</td>
          <td>25.959298</td>
          <td>0.133546</td>
          <td>25.664659</td>
          <td>0.193501</td>
          <td>25.242251</td>
          <td>0.292364</td>
          <td>0.097094</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.114925</td>
          <td>0.640810</td>
          <td>28.737418</td>
          <td>0.854975</td>
          <td>28.704466</td>
          <td>0.770894</td>
          <td>26.151135</td>
          <td>0.165923</td>
          <td>25.046594</td>
          <td>0.119819</td>
          <td>24.541081</td>
          <td>0.171785</td>
          <td>0.119666</td>
          <td>0.085456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.285866</td>
          <td>1.316537</td>
          <td>28.645159</td>
          <td>0.805363</td>
          <td>27.532946</td>
          <td>0.327036</td>
          <td>26.172599</td>
          <td>0.168912</td>
          <td>25.736324</td>
          <td>0.215721</td>
          <td>25.377240</td>
          <td>0.341595</td>
          <td>0.109217</td>
          <td>0.096802</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.591420</td>
          <td>0.824766</td>
          <td>26.341120</td>
          <td>0.121236</td>
          <td>25.999714</td>
          <td>0.079231</td>
          <td>25.681328</td>
          <td>0.097474</td>
          <td>25.566450</td>
          <td>0.166414</td>
          <td>25.212237</td>
          <td>0.266826</td>
          <td>0.022166</td>
          <td>0.016141</td>
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
          <td>27.558158</td>
          <td>0.876540</td>
          <td>26.176511</td>
          <td>0.120426</td>
          <td>25.395315</td>
          <td>0.054278</td>
          <td>25.102291</td>
          <td>0.068848</td>
          <td>24.862617</td>
          <td>0.105546</td>
          <td>24.519911</td>
          <td>0.174495</td>
          <td>0.128220</td>
          <td>0.108351</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.227476</td>
          <td>0.116837</td>
          <td>26.014744</td>
          <td>0.086240</td>
          <td>25.167153</td>
          <td>0.066750</td>
          <td>24.895225</td>
          <td>0.099860</td>
          <td>24.458664</td>
          <td>0.152215</td>
          <td>0.083154</td>
          <td>0.072578</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.236993</td>
          <td>0.403730</td>
          <td>27.409031</td>
          <td>0.409941</td>
          <td>26.036350</td>
          <td>0.121418</td>
          <td>26.053385</td>
          <td>0.200367</td>
          <td>25.774458</td>
          <td>0.287595</td>
          <td>25.511199</td>
          <td>0.483519</td>
          <td>0.225915</td>
          <td>0.203744</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.591553</td>
          <td>0.406269</td>
          <td>26.160165</td>
          <td>0.103693</td>
          <td>25.940922</td>
          <td>0.075336</td>
          <td>25.817988</td>
          <td>0.110027</td>
          <td>25.930192</td>
          <td>0.226356</td>
          <td>25.145811</td>
          <td>0.253080</td>
          <td>0.022843</td>
          <td>0.021318</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.957689</td>
          <td>1.775599</td>
          <td>26.839904</td>
          <td>0.192758</td>
          <td>26.912052</td>
          <td>0.182440</td>
          <td>26.408463</td>
          <td>0.190733</td>
          <td>25.883296</td>
          <td>0.226444</td>
          <td>25.333708</td>
          <td>0.306759</td>
          <td>0.067755</td>
          <td>0.053839</td>
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
