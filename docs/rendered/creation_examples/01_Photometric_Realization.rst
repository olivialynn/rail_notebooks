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

    <pzflow.flow.Flow at 0x7f62b63bb5e0>



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
    0      23.994413  0.004004  0.003213  
    1      25.391064  0.171360  0.113028  
    2      24.304707  0.143527  0.120366  
    3      25.291103  0.136555  0.113521  
    4      25.096743  0.054032  0.030376  
    ...          ...       ...       ...  
    99995  24.737946  0.037152  0.018752  
    99996  24.224169  0.202576  0.167316  
    99997  25.613836  0.054950  0.030800  
    99998  25.274899  0.055091  0.032167  
    99999  25.699642  0.145032  0.121242  
    
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
          <td>26.990915</td>
          <td>0.545115</td>
          <td>26.722790</td>
          <td>0.167615</td>
          <td>25.930023</td>
          <td>0.074120</td>
          <td>25.209293</td>
          <td>0.063921</td>
          <td>24.649521</td>
          <td>0.074519</td>
          <td>23.870340</td>
          <td>0.084326</td>
          <td>0.004004</td>
          <td>0.003213</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.316703</td>
          <td>1.266174</td>
          <td>26.917492</td>
          <td>0.197625</td>
          <td>26.440618</td>
          <td>0.116054</td>
          <td>26.283202</td>
          <td>0.163328</td>
          <td>25.907424</td>
          <td>0.220733</td>
          <td>25.269318</td>
          <td>0.278150</td>
          <td>0.171360</td>
          <td>0.113028</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.793263</td>
          <td>0.358590</td>
          <td>26.014552</td>
          <td>0.129640</td>
          <td>25.018262</td>
          <td>0.103080</td>
          <td>24.104507</td>
          <td>0.103578</td>
          <td>0.143527</td>
          <td>0.120366</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.720598</td>
          <td>1.559473</td>
          <td>27.630951</td>
          <td>0.353456</td>
          <td>27.412153</td>
          <td>0.264264</td>
          <td>26.138833</td>
          <td>0.144321</td>
          <td>25.556854</td>
          <td>0.164247</td>
          <td>25.169905</td>
          <td>0.256486</td>
          <td>0.136555</td>
          <td>0.113521</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.264802</td>
          <td>0.313257</td>
          <td>26.028960</td>
          <td>0.091910</td>
          <td>25.924927</td>
          <td>0.073786</td>
          <td>25.466602</td>
          <td>0.080262</td>
          <td>25.434605</td>
          <td>0.147926</td>
          <td>24.967138</td>
          <td>0.216899</td>
          <td>0.054032</td>
          <td>0.030376</td>
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
          <td>28.015740</td>
          <td>1.068281</td>
          <td>26.304786</td>
          <td>0.116958</td>
          <td>25.415545</td>
          <td>0.046965</td>
          <td>25.117278</td>
          <td>0.058911</td>
          <td>24.813942</td>
          <td>0.086153</td>
          <td>24.547244</td>
          <td>0.152032</td>
          <td>0.037152</td>
          <td>0.018752</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.611585</td>
          <td>0.833390</td>
          <td>26.774601</td>
          <td>0.175160</td>
          <td>26.032274</td>
          <td>0.081125</td>
          <td>25.361503</td>
          <td>0.073145</td>
          <td>24.793581</td>
          <td>0.084621</td>
          <td>24.838079</td>
          <td>0.194668</td>
          <td>0.202576</td>
          <td>0.167316</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.041798</td>
          <td>0.261632</td>
          <td>26.585331</td>
          <td>0.149036</td>
          <td>26.333623</td>
          <td>0.105711</td>
          <td>26.203581</td>
          <td>0.152574</td>
          <td>25.734311</td>
          <td>0.190929</td>
          <td>26.516592</td>
          <td>0.708544</td>
          <td>0.054950</td>
          <td>0.030800</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.409098</td>
          <td>0.351183</td>
          <td>26.393767</td>
          <td>0.126346</td>
          <td>26.016337</td>
          <td>0.079992</td>
          <td>25.799313</td>
          <td>0.107505</td>
          <td>25.842475</td>
          <td>0.209088</td>
          <td>25.342331</td>
          <td>0.295070</td>
          <td>0.055091</td>
          <td>0.032167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.650143</td>
          <td>0.854211</td>
          <td>26.352046</td>
          <td>0.121858</td>
          <td>26.596806</td>
          <td>0.132897</td>
          <td>26.360851</td>
          <td>0.174491</td>
          <td>26.076386</td>
          <td>0.253814</td>
          <td>25.628404</td>
          <td>0.370231</td>
          <td>0.145032</td>
          <td>0.121242</td>
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
          <td>26.539657</td>
          <td>0.430681</td>
          <td>26.765020</td>
          <td>0.199334</td>
          <td>26.017580</td>
          <td>0.094168</td>
          <td>25.115029</td>
          <td>0.069706</td>
          <td>24.784691</td>
          <td>0.098691</td>
          <td>23.907601</td>
          <td>0.102972</td>
          <td>0.004004</td>
          <td>0.003213</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.597872</td>
          <td>0.835628</td>
          <td>26.927293</td>
          <td>0.219375</td>
          <td>25.924653</td>
          <td>0.151359</td>
          <td>28.771263</td>
          <td>1.746542</td>
          <td>24.875648</td>
          <td>0.250868</td>
          <td>0.171360</td>
          <td>0.113028</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.094001</td>
          <td>0.589988</td>
          <td>28.213689</td>
          <td>0.591274</td>
          <td>25.897490</td>
          <td>0.146387</td>
          <td>24.980394</td>
          <td>0.123774</td>
          <td>24.436562</td>
          <td>0.172075</td>
          <td>0.143527</td>
          <td>0.120366</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.644122</td>
          <td>1.503191</td>
          <td>27.677217</td>
          <td>0.395355</td>
          <td>26.495196</td>
          <td>0.241051</td>
          <td>25.371481</td>
          <td>0.172288</td>
          <td>24.807391</td>
          <td>0.233653</td>
          <td>0.136555</td>
          <td>0.113521</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.371429</td>
          <td>0.380142</td>
          <td>25.964512</td>
          <td>0.100785</td>
          <td>25.960110</td>
          <td>0.090112</td>
          <td>25.886379</td>
          <td>0.137866</td>
          <td>25.436666</td>
          <td>0.174457</td>
          <td>25.013867</td>
          <td>0.265329</td>
          <td>0.054032</td>
          <td>0.030376</td>
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
          <td>28.792592</td>
          <td>1.721376</td>
          <td>26.228657</td>
          <td>0.126429</td>
          <td>25.377308</td>
          <td>0.053631</td>
          <td>25.068101</td>
          <td>0.067072</td>
          <td>25.065776</td>
          <td>0.126453</td>
          <td>24.580693</td>
          <td>0.184437</td>
          <td>0.037152</td>
          <td>0.018752</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.655147</td>
          <td>0.199136</td>
          <td>26.007939</td>
          <td>0.103677</td>
          <td>25.156118</td>
          <td>0.080601</td>
          <td>24.944638</td>
          <td>0.125891</td>
          <td>24.059695</td>
          <td>0.130747</td>
          <td>0.202576</td>
          <td>0.167316</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.545757</td>
          <td>0.434574</td>
          <td>26.849365</td>
          <td>0.215137</td>
          <td>26.442683</td>
          <td>0.137261</td>
          <td>26.426856</td>
          <td>0.218177</td>
          <td>25.715704</td>
          <td>0.220642</td>
          <td>25.953172</td>
          <td>0.548501</td>
          <td>0.054950</td>
          <td>0.030800</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.270160</td>
          <td>0.351344</td>
          <td>26.231237</td>
          <td>0.127157</td>
          <td>26.132670</td>
          <td>0.104870</td>
          <td>25.595634</td>
          <td>0.107151</td>
          <td>26.353974</td>
          <td>0.369556</td>
          <td>24.825072</td>
          <td>0.227222</td>
          <td>0.055091</td>
          <td>0.032167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.993761</td>
          <td>0.622491</td>
          <td>26.535832</td>
          <td>0.172658</td>
          <td>26.506136</td>
          <td>0.152312</td>
          <td>26.370924</td>
          <td>0.218808</td>
          <td>25.996434</td>
          <td>0.291163</td>
          <td>25.711865</td>
          <td>0.479830</td>
          <td>0.145032</td>
          <td>0.121242</td>
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
          <td>26.884110</td>
          <td>0.504298</td>
          <td>26.681271</td>
          <td>0.161811</td>
          <td>26.102938</td>
          <td>0.086354</td>
          <td>25.129984</td>
          <td>0.059590</td>
          <td>24.614288</td>
          <td>0.072246</td>
          <td>24.004414</td>
          <td>0.094897</td>
          <td>0.004004</td>
          <td>0.003213</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.500739</td>
          <td>0.379857</td>
          <td>26.479135</td>
          <td>0.148882</td>
          <td>26.506824</td>
          <td>0.244933</td>
          <td>25.935073</td>
          <td>0.277154</td>
          <td>25.170977</td>
          <td>0.316052</td>
          <td>0.171360</td>
          <td>0.113028</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.456767</td>
          <td>1.351411</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.978339</td>
          <td>0.152975</td>
          <td>25.124919</td>
          <td>0.136832</td>
          <td>24.200063</td>
          <td>0.137032</td>
          <td>0.143527</td>
          <td>0.120366</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.614660</td>
          <td>0.400751</td>
          <td>27.743972</td>
          <td>0.403269</td>
          <td>26.160489</td>
          <td>0.175778</td>
          <td>25.602049</td>
          <td>0.202198</td>
          <td>25.479829</td>
          <td>0.387456</td>
          <td>0.136555</td>
          <td>0.113521</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.352816</td>
          <td>0.341321</td>
          <td>26.196897</td>
          <td>0.108819</td>
          <td>25.965808</td>
          <td>0.078471</td>
          <td>25.741817</td>
          <td>0.104968</td>
          <td>25.418463</td>
          <td>0.149532</td>
          <td>24.477397</td>
          <td>0.146900</td>
          <td>0.054032</td>
          <td>0.030376</td>
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
          <td>27.645809</td>
          <td>0.856839</td>
          <td>26.548864</td>
          <td>0.145870</td>
          <td>25.435556</td>
          <td>0.048371</td>
          <td>25.043742</td>
          <td>0.055872</td>
          <td>24.642549</td>
          <td>0.074929</td>
          <td>24.633979</td>
          <td>0.165664</td>
          <td>0.037152</td>
          <td>0.018752</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.760864</td>
          <td>0.565188</td>
          <td>26.816678</td>
          <td>0.239757</td>
          <td>25.797906</td>
          <td>0.091425</td>
          <td>25.373723</td>
          <td>0.103603</td>
          <td>25.037387</td>
          <td>0.144438</td>
          <td>24.096780</td>
          <td>0.143133</td>
          <td>0.202576</td>
          <td>0.167316</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.989117</td>
          <td>1.064584</td>
          <td>26.626733</td>
          <td>0.157869</td>
          <td>26.497262</td>
          <td>0.125094</td>
          <td>26.313854</td>
          <td>0.172148</td>
          <td>25.932598</td>
          <td>0.231008</td>
          <td>25.383284</td>
          <td>0.312558</td>
          <td>0.054950</td>
          <td>0.030800</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.150315</td>
          <td>0.619975</td>
          <td>26.076042</td>
          <td>0.098037</td>
          <td>26.213752</td>
          <td>0.097749</td>
          <td>25.756878</td>
          <td>0.106521</td>
          <td>26.297582</td>
          <td>0.311220</td>
          <td>24.970796</td>
          <td>0.223354</td>
          <td>0.055091</td>
          <td>0.032167</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.479664</td>
          <td>0.420891</td>
          <td>26.580620</td>
          <td>0.175733</td>
          <td>26.616854</td>
          <td>0.163694</td>
          <td>27.022608</td>
          <td>0.362662</td>
          <td>26.262372</td>
          <td>0.352385</td>
          <td>25.144773</td>
          <td>0.302769</td>
          <td>0.145032</td>
          <td>0.121242</td>
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
