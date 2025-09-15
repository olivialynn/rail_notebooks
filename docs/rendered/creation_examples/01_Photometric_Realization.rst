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

    <pzflow.flow.Flow at 0x7fe480a626b0>



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
    0      23.994413  0.198579  0.190853  
    1      25.391064  0.104357  0.052560  
    2      24.304707  0.050880  0.032448  
    3      25.291103  0.112422  0.091038  
    4      25.096743  0.194521  0.178755  
    ...          ...       ...       ...  
    99995  24.737946  0.045201  0.030469  
    99996  24.224169  0.165795  0.121958  
    99997  25.613836  0.215999  0.147088  
    99998  25.274899  0.126528  0.080056  
    99999  25.699642  0.006663  0.005158  
    
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
          <td>27.274015</td>
          <td>0.665700</td>
          <td>26.763080</td>
          <td>0.173456</td>
          <td>26.249857</td>
          <td>0.098236</td>
          <td>25.148190</td>
          <td>0.060549</td>
          <td>24.639600</td>
          <td>0.073868</td>
          <td>24.047403</td>
          <td>0.098525</td>
          <td>0.198579</td>
          <td>0.190853</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.536331</td>
          <td>0.328006</td>
          <td>26.765476</td>
          <td>0.153666</td>
          <td>26.176915</td>
          <td>0.149123</td>
          <td>25.957025</td>
          <td>0.230018</td>
          <td>25.517584</td>
          <td>0.339375</td>
          <td>0.104357</td>
          <td>0.052560</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>32.453676</td>
          <td>5.008340</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.487510</td>
          <td>0.602341</td>
          <td>26.182994</td>
          <td>0.149903</td>
          <td>25.144960</td>
          <td>0.115135</td>
          <td>24.208231</td>
          <td>0.113397</td>
          <td>0.050880</td>
          <td>0.032448</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.269032</td>
          <td>0.571175</td>
          <td>27.931105</td>
          <td>0.399138</td>
          <td>26.485825</td>
          <td>0.193946</td>
          <td>25.478062</td>
          <td>0.153546</td>
          <td>25.847155</td>
          <td>0.438048</td>
          <td>0.112422</td>
          <td>0.091038</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.080585</td>
          <td>0.270033</td>
          <td>26.125893</td>
          <td>0.100059</td>
          <td>26.004668</td>
          <td>0.079172</td>
          <td>25.633564</td>
          <td>0.092974</td>
          <td>25.679846</td>
          <td>0.182342</td>
          <td>25.125664</td>
          <td>0.247336</td>
          <td>0.194521</td>
          <td>0.178755</td>
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
          <td>27.440583</td>
          <td>0.745168</td>
          <td>26.458606</td>
          <td>0.133633</td>
          <td>25.477550</td>
          <td>0.049623</td>
          <td>25.120977</td>
          <td>0.059105</td>
          <td>24.828438</td>
          <td>0.087259</td>
          <td>24.556770</td>
          <td>0.153278</td>
          <td>0.045201</td>
          <td>0.030469</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.599721</td>
          <td>0.407195</td>
          <td>26.441871</td>
          <td>0.131715</td>
          <td>26.085091</td>
          <td>0.084991</td>
          <td>25.181153</td>
          <td>0.062346</td>
          <td>24.872650</td>
          <td>0.090720</td>
          <td>24.161279</td>
          <td>0.108846</td>
          <td>0.165795</td>
          <td>0.121958</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.809497</td>
          <td>0.477153</td>
          <td>26.473160</td>
          <td>0.135323</td>
          <td>26.420763</td>
          <td>0.114065</td>
          <td>26.262093</td>
          <td>0.160410</td>
          <td>25.905165</td>
          <td>0.220318</td>
          <td>25.460197</td>
          <td>0.324276</td>
          <td>0.215999</td>
          <td>0.147088</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.493351</td>
          <td>0.375084</td>
          <td>26.219779</td>
          <td>0.108613</td>
          <td>26.043392</td>
          <td>0.081924</td>
          <td>25.938521</td>
          <td>0.121368</td>
          <td>25.916362</td>
          <td>0.222381</td>
          <td>25.285400</td>
          <td>0.281802</td>
          <td>0.126528</td>
          <td>0.080056</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.677932</td>
          <td>0.161327</td>
          <td>26.400431</td>
          <td>0.112061</td>
          <td>26.351780</td>
          <td>0.173151</td>
          <td>25.961023</td>
          <td>0.230782</td>
          <td>26.551455</td>
          <td>0.725390</td>
          <td>0.006663</td>
          <td>0.005158</td>
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
          <td>28.196955</td>
          <td>1.346671</td>
          <td>26.545822</td>
          <td>0.183152</td>
          <td>26.064881</td>
          <td>0.110005</td>
          <td>25.112997</td>
          <td>0.078365</td>
          <td>24.894166</td>
          <td>0.121642</td>
          <td>24.113788</td>
          <td>0.138323</td>
          <td>0.198579</td>
          <td>0.190853</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.989954</td>
          <td>0.533948</td>
          <td>26.144233</td>
          <td>0.107600</td>
          <td>26.315492</td>
          <td>0.201899</td>
          <td>25.474075</td>
          <td>0.182884</td>
          <td>25.333406</td>
          <td>0.348054</td>
          <td>0.104357</td>
          <td>0.052560</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.372606</td>
          <td>0.780860</td>
          <td>28.517783</td>
          <td>0.761640</td>
          <td>27.825757</td>
          <td>0.426099</td>
          <td>25.892382</td>
          <td>0.138535</td>
          <td>25.051632</td>
          <td>0.125309</td>
          <td>24.404590</td>
          <td>0.159302</td>
          <td>0.050880</td>
          <td>0.032448</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.287877</td>
          <td>0.665039</td>
          <td>28.264690</td>
          <td>0.602300</td>
          <td>26.189102</td>
          <td>0.183643</td>
          <td>25.659985</td>
          <td>0.216224</td>
          <td>25.267639</td>
          <td>0.334057</td>
          <td>0.112422</td>
          <td>0.091038</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.260327</td>
          <td>0.373527</td>
          <td>26.070841</td>
          <td>0.121003</td>
          <td>25.965869</td>
          <td>0.100060</td>
          <td>25.702693</td>
          <td>0.130156</td>
          <td>25.463782</td>
          <td>0.196448</td>
          <td>25.175318</td>
          <td>0.331769</td>
          <td>0.194521</td>
          <td>0.178755</td>
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
          <td>26.560997</td>
          <td>0.439136</td>
          <td>26.575020</td>
          <td>0.170510</td>
          <td>25.431335</td>
          <td>0.056383</td>
          <td>25.036430</td>
          <td>0.065360</td>
          <td>24.765268</td>
          <td>0.097510</td>
          <td>24.450699</td>
          <td>0.165510</td>
          <td>0.045201</td>
          <td>0.030469</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.911129</td>
          <td>0.590853</td>
          <td>26.927030</td>
          <td>0.241592</td>
          <td>26.206960</td>
          <td>0.118757</td>
          <td>25.283746</td>
          <td>0.086718</td>
          <td>24.757568</td>
          <td>0.103015</td>
          <td>24.547774</td>
          <td>0.191052</td>
          <td>0.165795</td>
          <td>0.121958</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.574416</td>
          <td>0.937827</td>
          <td>26.435553</td>
          <td>0.165115</td>
          <td>26.332724</td>
          <td>0.137217</td>
          <td>26.456299</td>
          <td>0.245483</td>
          <td>25.585038</td>
          <td>0.216775</td>
          <td>26.839627</td>
          <td>1.057489</td>
          <td>0.215999</td>
          <td>0.147088</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.766657</td>
          <td>0.239155</td>
          <td>26.373402</td>
          <td>0.147574</td>
          <td>26.090392</td>
          <td>0.104094</td>
          <td>26.047199</td>
          <td>0.163137</td>
          <td>25.686350</td>
          <td>0.221479</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.126528</td>
          <td>0.080056</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.169130</td>
          <td>0.678915</td>
          <td>26.433725</td>
          <td>0.150486</td>
          <td>26.604821</td>
          <td>0.156767</td>
          <td>26.182997</td>
          <td>0.176565</td>
          <td>26.034772</td>
          <td>0.284974</td>
          <td>26.166992</td>
          <td>0.634956</td>
          <td>0.006663</td>
          <td>0.005158</td>
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
          <td>26.002287</td>
          <td>0.323632</td>
          <td>26.757178</td>
          <td>0.233243</td>
          <td>26.186793</td>
          <td>0.131537</td>
          <td>25.059704</td>
          <td>0.080670</td>
          <td>24.641007</td>
          <td>0.104969</td>
          <td>23.957864</td>
          <td>0.130174</td>
          <td>0.198579</td>
          <td>0.190853</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.702778</td>
          <td>0.398558</td>
          <td>26.546796</td>
          <td>0.138042</td>
          <td>26.304541</td>
          <td>0.180798</td>
          <td>25.835739</td>
          <td>0.224837</td>
          <td>25.235858</td>
          <td>0.292880</td>
          <td>0.104357</td>
          <td>0.052560</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.358581</td>
          <td>1.917238</td>
          <td>27.877011</td>
          <td>0.390979</td>
          <td>25.986265</td>
          <td>0.129687</td>
          <td>24.934431</td>
          <td>0.098104</td>
          <td>24.389114</td>
          <td>0.135974</td>
          <td>0.050880</td>
          <td>0.032448</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.946191</td>
          <td>1.088329</td>
          <td>29.130763</td>
          <td>1.084829</td>
          <td>27.255459</td>
          <td>0.260902</td>
          <td>25.972378</td>
          <td>0.141965</td>
          <td>25.645340</td>
          <td>0.199464</td>
          <td>25.653527</td>
          <td>0.422431</td>
          <td>0.112422</td>
          <td>0.091038</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.104322</td>
          <td>0.345545</td>
          <td>26.316757</td>
          <td>0.158099</td>
          <td>25.940874</td>
          <td>0.104071</td>
          <td>25.984610</td>
          <td>0.176238</td>
          <td>25.433240</td>
          <td>0.203030</td>
          <td>25.522280</td>
          <td>0.458300</td>
          <td>0.194521</td>
          <td>0.178755</td>
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
          <td>26.341173</td>
          <td>0.122779</td>
          <td>25.465298</td>
          <td>0.050077</td>
          <td>25.051826</td>
          <td>0.056764</td>
          <td>24.916148</td>
          <td>0.096130</td>
          <td>24.878142</td>
          <td>0.205312</td>
          <td>0.045201</td>
          <td>0.030469</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.876588</td>
          <td>0.230574</td>
          <td>26.017321</td>
          <td>0.100040</td>
          <td>25.270740</td>
          <td>0.085187</td>
          <td>24.754838</td>
          <td>0.102150</td>
          <td>24.052251</td>
          <td>0.124248</td>
          <td>0.165795</td>
          <td>0.121958</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.047378</td>
          <td>0.327632</td>
          <td>26.586436</td>
          <td>0.196552</td>
          <td>26.403375</td>
          <td>0.153414</td>
          <td>26.968639</td>
          <td>0.388151</td>
          <td>25.603438</td>
          <td>0.231188</td>
          <td>25.333766</td>
          <td>0.392531</td>
          <td>0.215999</td>
          <td>0.147088</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.199398</td>
          <td>1.258871</td>
          <td>26.200500</td>
          <td>0.119658</td>
          <td>26.192075</td>
          <td>0.106283</td>
          <td>26.051801</td>
          <td>0.152863</td>
          <td>25.443699</td>
          <td>0.169119</td>
          <td>24.682162</td>
          <td>0.194123</td>
          <td>0.126528</td>
          <td>0.080056</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.688058</td>
          <td>0.875224</td>
          <td>26.491945</td>
          <td>0.137590</td>
          <td>26.587690</td>
          <td>0.131917</td>
          <td>26.334225</td>
          <td>0.170670</td>
          <td>25.724972</td>
          <td>0.189518</td>
          <td>26.045127</td>
          <td>0.508035</td>
          <td>0.006663</td>
          <td>0.005158</td>
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
