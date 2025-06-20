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

    <pzflow.flow.Flow at 0x7f5d8d8ad990>



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
    0      23.994413  0.108585  0.095371  
    1      25.391064  0.035497  0.034561  
    2      24.304707  0.106454  0.083670  
    3      25.291103  0.078391  0.068923  
    4      25.096743  0.126065  0.073778  
    ...          ...       ...       ...  
    99995  24.737946  0.108973  0.058852  
    99996  24.224169  0.091623  0.087996  
    99997  25.613836  0.030814  0.030141  
    99998  25.274899  0.065160  0.040314  
    99999  25.699642  0.077109  0.073039  
    
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
          <td>27.284493</td>
          <td>0.670511</td>
          <td>26.894409</td>
          <td>0.193825</td>
          <td>25.972794</td>
          <td>0.076975</td>
          <td>25.238625</td>
          <td>0.065605</td>
          <td>24.870067</td>
          <td>0.090514</td>
          <td>24.003126</td>
          <td>0.094772</td>
          <td>0.108585</td>
          <td>0.095371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.151042</td>
          <td>1.154920</td>
          <td>27.423027</td>
          <td>0.299614</td>
          <td>26.645162</td>
          <td>0.138564</td>
          <td>26.334611</td>
          <td>0.170642</td>
          <td>25.511992</td>
          <td>0.158072</td>
          <td>25.905837</td>
          <td>0.457875</td>
          <td>0.035497</td>
          <td>0.034561</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.359274</td>
          <td>1.295651</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.939467</td>
          <td>0.401715</td>
          <td>26.038992</td>
          <td>0.132411</td>
          <td>25.169505</td>
          <td>0.117622</td>
          <td>24.371189</td>
          <td>0.130636</td>
          <td>0.106454</td>
          <td>0.083670</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.003893</td>
          <td>0.550251</td>
          <td>28.041413</td>
          <td>0.483791</td>
          <td>27.568110</td>
          <td>0.299872</td>
          <td>25.998040</td>
          <td>0.127800</td>
          <td>25.575251</td>
          <td>0.166843</td>
          <td>24.828603</td>
          <td>0.193121</td>
          <td>0.078391</td>
          <td>0.068923</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.376610</td>
          <td>0.342318</td>
          <td>26.084997</td>
          <td>0.096539</td>
          <td>26.090050</td>
          <td>0.085364</td>
          <td>25.530180</td>
          <td>0.084890</td>
          <td>25.652737</td>
          <td>0.178202</td>
          <td>25.443074</td>
          <td>0.319885</td>
          <td>0.126065</td>
          <td>0.073778</td>
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
          <td>26.285638</td>
          <td>0.318506</td>
          <td>26.083623</td>
          <td>0.096423</td>
          <td>25.495241</td>
          <td>0.050409</td>
          <td>25.002555</td>
          <td>0.053207</td>
          <td>24.869016</td>
          <td>0.090431</td>
          <td>24.769660</td>
          <td>0.183747</td>
          <td>0.108973</td>
          <td>0.058852</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.122450</td>
          <td>0.279367</td>
          <td>26.700434</td>
          <td>0.164453</td>
          <td>26.085569</td>
          <td>0.085027</td>
          <td>25.177766</td>
          <td>0.062159</td>
          <td>24.840129</td>
          <td>0.088162</td>
          <td>24.375441</td>
          <td>0.131118</td>
          <td>0.091623</td>
          <td>0.087996</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.376988</td>
          <td>0.342420</td>
          <td>26.550891</td>
          <td>0.144692</td>
          <td>26.376109</td>
          <td>0.109708</td>
          <td>26.518231</td>
          <td>0.199305</td>
          <td>25.632345</td>
          <td>0.175146</td>
          <td>25.699921</td>
          <td>0.391370</td>
          <td>0.030814</td>
          <td>0.030141</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.425623</td>
          <td>0.355766</td>
          <td>26.182815</td>
          <td>0.105164</td>
          <td>26.149288</td>
          <td>0.089933</td>
          <td>25.931981</td>
          <td>0.120680</td>
          <td>25.726392</td>
          <td>0.189658</td>
          <td>25.331451</td>
          <td>0.292493</td>
          <td>0.065160</td>
          <td>0.040314</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.350318</td>
          <td>0.701312</td>
          <td>26.644980</td>
          <td>0.156848</td>
          <td>26.541851</td>
          <td>0.126723</td>
          <td>26.287695</td>
          <td>0.163955</td>
          <td>26.164776</td>
          <td>0.272823</td>
          <td>25.159969</td>
          <td>0.254405</td>
          <td>0.077109</td>
          <td>0.073039</td>
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
          <td>26.320161</td>
          <td>0.372258</td>
          <td>26.541103</td>
          <td>0.169969</td>
          <td>26.168619</td>
          <td>0.111208</td>
          <td>25.125535</td>
          <td>0.072917</td>
          <td>24.807973</td>
          <td>0.104222</td>
          <td>23.942110</td>
          <td>0.109907</td>
          <td>0.108585</td>
          <td>0.095371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.889984</td>
          <td>2.668041</td>
          <td>26.955036</td>
          <td>0.234359</td>
          <td>26.584609</td>
          <td>0.154687</td>
          <td>26.233250</td>
          <td>0.184987</td>
          <td>26.090557</td>
          <td>0.299200</td>
          <td>25.789683</td>
          <td>0.485529</td>
          <td>0.035497</td>
          <td>0.034561</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.227030</td>
          <td>0.635715</td>
          <td>27.664243</td>
          <td>0.384368</td>
          <td>26.034131</td>
          <td>0.160311</td>
          <td>24.989466</td>
          <td>0.121584</td>
          <td>24.226830</td>
          <td>0.140105</td>
          <td>0.106454</td>
          <td>0.083670</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.433355</td>
          <td>1.455866</td>
          <td>28.152246</td>
          <td>0.597970</td>
          <td>27.204306</td>
          <td>0.263536</td>
          <td>26.136194</td>
          <td>0.172784</td>
          <td>25.417892</td>
          <td>0.173652</td>
          <td>25.613497</td>
          <td>0.430619</td>
          <td>0.078391</td>
          <td>0.068923</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.392658</td>
          <td>0.393897</td>
          <td>26.249990</td>
          <td>0.132493</td>
          <td>26.005757</td>
          <td>0.096488</td>
          <td>25.803979</td>
          <td>0.132140</td>
          <td>25.623209</td>
          <td>0.209767</td>
          <td>25.194502</td>
          <td>0.315296</td>
          <td>0.126065</td>
          <td>0.073778</td>
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
          <td>26.284185</td>
          <td>0.135308</td>
          <td>25.409543</td>
          <td>0.056448</td>
          <td>25.269442</td>
          <td>0.082018</td>
          <td>24.933798</td>
          <td>0.115273</td>
          <td>24.687032</td>
          <td>0.206181</td>
          <td>0.108973</td>
          <td>0.058852</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.619588</td>
          <td>0.924653</td>
          <td>26.538211</td>
          <td>0.168427</td>
          <td>26.082540</td>
          <td>0.102379</td>
          <td>25.107779</td>
          <td>0.071217</td>
          <td>24.904580</td>
          <td>0.112542</td>
          <td>24.188918</td>
          <td>0.135123</td>
          <td>0.091623</td>
          <td>0.087996</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.648568</td>
          <td>0.929513</td>
          <td>26.482364</td>
          <td>0.157303</td>
          <td>26.505785</td>
          <td>0.144425</td>
          <td>26.207925</td>
          <td>0.180883</td>
          <td>25.770307</td>
          <td>0.230107</td>
          <td>24.792537</td>
          <td>0.220362</td>
          <td>0.030814</td>
          <td>0.030141</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.836225</td>
          <td>0.540140</td>
          <td>26.137144</td>
          <td>0.117509</td>
          <td>26.043149</td>
          <td>0.097256</td>
          <td>25.903229</td>
          <td>0.140366</td>
          <td>25.741143</td>
          <td>0.226036</td>
          <td>25.216690</td>
          <td>0.313573</td>
          <td>0.065160</td>
          <td>0.040314</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.722788</td>
          <td>0.980874</td>
          <td>26.924995</td>
          <td>0.231480</td>
          <td>26.674076</td>
          <td>0.169365</td>
          <td>25.980772</td>
          <td>0.151410</td>
          <td>26.138393</td>
          <td>0.315081</td>
          <td>26.344826</td>
          <td>0.727916</td>
          <td>0.077109</td>
          <td>0.073039</td>
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
          <td>27.212187</td>
          <td>0.684038</td>
          <td>26.669818</td>
          <td>0.178110</td>
          <td>26.044011</td>
          <td>0.092833</td>
          <td>25.310028</td>
          <td>0.079634</td>
          <td>24.627451</td>
          <td>0.082790</td>
          <td>23.854136</td>
          <td>0.094548</td>
          <td>0.108585</td>
          <td>0.095371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.549415</td>
          <td>0.335711</td>
          <td>26.783150</td>
          <td>0.158521</td>
          <td>26.658894</td>
          <td>0.227845</td>
          <td>25.886779</td>
          <td>0.220365</td>
          <td>25.316790</td>
          <td>0.293609</td>
          <td>0.035497</td>
          <td>0.034561</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.203563</td>
          <td>0.537608</td>
          <td>25.810565</td>
          <td>0.121693</td>
          <td>24.926359</td>
          <td>0.106140</td>
          <td>24.304127</td>
          <td>0.137915</td>
          <td>0.106454</td>
          <td>0.083670</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.575790</td>
          <td>0.321221</td>
          <td>26.226485</td>
          <td>0.166880</td>
          <td>25.298591</td>
          <td>0.140751</td>
          <td>26.583087</td>
          <td>0.781767</td>
          <td>0.078391</td>
          <td>0.068923</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.694931</td>
          <td>0.213752</td>
          <td>26.169220</td>
          <td>0.115822</td>
          <td>25.932439</td>
          <td>0.084106</td>
          <td>25.799266</td>
          <td>0.122158</td>
          <td>25.121442</td>
          <td>0.127452</td>
          <td>25.017450</td>
          <td>0.254986</td>
          <td>0.126065</td>
          <td>0.073778</td>
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
          <td>27.098238</td>
          <td>0.620709</td>
          <td>26.542826</td>
          <td>0.155517</td>
          <td>25.489793</td>
          <td>0.055066</td>
          <td>25.137724</td>
          <td>0.066111</td>
          <td>24.737663</td>
          <td>0.088337</td>
          <td>24.564075</td>
          <td>0.169198</td>
          <td>0.108973</td>
          <td>0.058852</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.221693</td>
          <td>0.678515</td>
          <td>26.575508</td>
          <td>0.160820</td>
          <td>26.056148</td>
          <td>0.091476</td>
          <td>25.146551</td>
          <td>0.067111</td>
          <td>24.800438</td>
          <td>0.093979</td>
          <td>24.175086</td>
          <td>0.121920</td>
          <td>0.091623</td>
          <td>0.087996</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.201653</td>
          <td>0.300173</td>
          <td>26.987506</td>
          <td>0.211738</td>
          <td>26.470008</td>
          <td>0.120539</td>
          <td>26.212063</td>
          <td>0.155661</td>
          <td>25.716254</td>
          <td>0.190304</td>
          <td>25.734853</td>
          <td>0.406634</td>
          <td>0.030814</td>
          <td>0.030141</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.433076</td>
          <td>0.366456</td>
          <td>26.006150</td>
          <td>0.093130</td>
          <td>26.142224</td>
          <td>0.092847</td>
          <td>25.813557</td>
          <td>0.113250</td>
          <td>25.430924</td>
          <td>0.153036</td>
          <td>25.430267</td>
          <td>0.328161</td>
          <td>0.065160</td>
          <td>0.040314</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.389560</td>
          <td>0.361722</td>
          <td>26.627024</td>
          <td>0.164094</td>
          <td>26.614641</td>
          <td>0.144714</td>
          <td>26.264275</td>
          <td>0.172738</td>
          <td>25.996127</td>
          <td>0.253927</td>
          <td>25.066278</td>
          <td>0.252360</td>
          <td>0.077109</td>
          <td>0.073039</td>
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
