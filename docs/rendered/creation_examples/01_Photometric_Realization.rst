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

    <pzflow.flow.Flow at 0x7f31ddabe0b0>



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
    0      23.994413  0.220823  0.217676  
    1      25.391064  0.147223  0.120762  
    2      24.304707  0.103529  0.058534  
    3      25.291103  0.073886  0.063951  
    4      25.096743  0.017565  0.015458  
    ...          ...       ...       ...  
    99995  24.737946  0.104253  0.065162  
    99996  24.224169  0.048120  0.039219  
    99997  25.613836  0.202512  0.171009  
    99998  25.274899  0.084949  0.053185  
    99999  25.699642  0.024513  0.018858  
    
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
          <td>27.588240</td>
          <td>0.820950</td>
          <td>26.586583</td>
          <td>0.149196</td>
          <td>26.086315</td>
          <td>0.085083</td>
          <td>25.113351</td>
          <td>0.058706</td>
          <td>24.772168</td>
          <td>0.083040</td>
          <td>24.051618</td>
          <td>0.098890</td>
          <td>0.220823</td>
          <td>0.217676</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.637813</td>
          <td>0.847516</td>
          <td>27.620958</td>
          <td>0.350691</td>
          <td>26.614759</td>
          <td>0.134975</td>
          <td>26.212275</td>
          <td>0.153715</td>
          <td>26.043069</td>
          <td>0.246960</td>
          <td>25.475202</td>
          <td>0.328168</td>
          <td>0.147223</td>
          <td>0.120762</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.392681</td>
          <td>0.562983</td>
          <td>25.886670</td>
          <td>0.116016</td>
          <td>25.070275</td>
          <td>0.107875</td>
          <td>24.365001</td>
          <td>0.129938</td>
          <td>0.103529</td>
          <td>0.058534</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.621070</td>
          <td>0.838480</td>
          <td>27.764481</td>
          <td>0.392202</td>
          <td>27.211671</td>
          <td>0.224018</td>
          <td>25.953659</td>
          <td>0.122974</td>
          <td>25.908715</td>
          <td>0.220970</td>
          <td>25.063779</td>
          <td>0.235025</td>
          <td>0.073886</td>
          <td>0.063951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.154038</td>
          <td>0.286597</td>
          <td>26.202321</td>
          <td>0.106971</td>
          <td>25.974447</td>
          <td>0.077087</td>
          <td>25.827886</td>
          <td>0.110221</td>
          <td>25.347171</td>
          <td>0.137199</td>
          <td>25.301909</td>
          <td>0.285595</td>
          <td>0.017565</td>
          <td>0.015458</td>
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
          <td>26.860697</td>
          <td>0.495617</td>
          <td>26.528442</td>
          <td>0.141925</td>
          <td>25.469184</td>
          <td>0.049256</td>
          <td>25.115769</td>
          <td>0.058832</td>
          <td>24.871435</td>
          <td>0.090623</td>
          <td>24.660615</td>
          <td>0.167501</td>
          <td>0.104253</td>
          <td>0.065162</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.934373</td>
          <td>0.523170</td>
          <td>26.691521</td>
          <td>0.163208</td>
          <td>26.065839</td>
          <td>0.083562</td>
          <td>25.096764</td>
          <td>0.057848</td>
          <td>24.877479</td>
          <td>0.091106</td>
          <td>24.301448</td>
          <td>0.122974</td>
          <td>0.048120</td>
          <td>0.039219</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.994237</td>
          <td>0.546426</td>
          <td>26.723890</td>
          <td>0.167772</td>
          <td>26.293372</td>
          <td>0.102054</td>
          <td>26.718446</td>
          <td>0.235515</td>
          <td>26.023724</td>
          <td>0.243057</td>
          <td>25.788441</td>
          <td>0.418915</td>
          <td>0.202512</td>
          <td>0.171009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.937222</td>
          <td>0.524259</td>
          <td>26.198263</td>
          <td>0.106592</td>
          <td>26.024305</td>
          <td>0.080556</td>
          <td>25.937184</td>
          <td>0.121227</td>
          <td>25.528821</td>
          <td>0.160362</td>
          <td>25.701194</td>
          <td>0.391756</td>
          <td>0.084949</td>
          <td>0.053185</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.498048</td>
          <td>0.774064</td>
          <td>26.807369</td>
          <td>0.180093</td>
          <td>26.714838</td>
          <td>0.147131</td>
          <td>26.181050</td>
          <td>0.149653</td>
          <td>25.580543</td>
          <td>0.167597</td>
          <td>25.476902</td>
          <td>0.328611</td>
          <td>0.024513</td>
          <td>0.018858</td>
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
          <td>26.105191</td>
          <td>0.339111</td>
          <td>27.029499</td>
          <td>0.279860</td>
          <td>26.030352</td>
          <td>0.109588</td>
          <td>25.318815</td>
          <td>0.096515</td>
          <td>24.572700</td>
          <td>0.094329</td>
          <td>23.859855</td>
          <td>0.114006</td>
          <td>0.220823</td>
          <td>0.217676</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.652211</td>
          <td>0.959545</td>
          <td>27.075779</td>
          <td>0.270839</td>
          <td>26.727636</td>
          <td>0.184066</td>
          <td>26.341673</td>
          <td>0.213696</td>
          <td>25.653263</td>
          <td>0.219889</td>
          <td>24.737987</td>
          <td>0.222110</td>
          <td>0.147223</td>
          <td>0.120762</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.375924</td>
          <td>2.068493</td>
          <td>28.178703</td>
          <td>0.561475</td>
          <td>25.862948</td>
          <td>0.137444</td>
          <td>25.373645</td>
          <td>0.168076</td>
          <td>24.244737</td>
          <td>0.141305</td>
          <td>0.103529</td>
          <td>0.058534</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.685553</td>
          <td>2.491948</td>
          <td>30.892405</td>
          <td>2.515152</td>
          <td>27.214281</td>
          <td>0.265145</td>
          <td>25.971832</td>
          <td>0.149821</td>
          <td>25.307134</td>
          <td>0.157670</td>
          <td>24.878802</td>
          <td>0.239661</td>
          <td>0.073886</td>
          <td>0.063951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.809304</td>
          <td>0.526697</td>
          <td>26.323456</td>
          <td>0.136975</td>
          <td>25.930104</td>
          <td>0.087277</td>
          <td>25.792237</td>
          <td>0.126370</td>
          <td>25.336019</td>
          <td>0.159261</td>
          <td>25.084712</td>
          <td>0.279591</td>
          <td>0.017565</td>
          <td>0.015458</td>
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
          <td>26.217246</td>
          <td>0.341228</td>
          <td>26.363905</td>
          <td>0.144882</td>
          <td>25.399293</td>
          <td>0.055921</td>
          <td>25.085398</td>
          <td>0.069692</td>
          <td>24.770286</td>
          <td>0.099907</td>
          <td>24.769669</td>
          <td>0.220845</td>
          <td>0.104253</td>
          <td>0.065162</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.016094</td>
          <td>2.785779</td>
          <td>27.232323</td>
          <td>0.294491</td>
          <td>25.953487</td>
          <td>0.089592</td>
          <td>25.157823</td>
          <td>0.072886</td>
          <td>24.960364</td>
          <td>0.115799</td>
          <td>24.415004</td>
          <td>0.160784</td>
          <td>0.048120</td>
          <td>0.039219</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.615460</td>
          <td>0.963582</td>
          <td>26.492912</td>
          <td>0.173918</td>
          <td>26.361952</td>
          <td>0.141223</td>
          <td>26.220795</td>
          <td>0.202563</td>
          <td>25.868121</td>
          <td>0.274625</td>
          <td>25.013015</td>
          <td>0.291487</td>
          <td>0.202512</td>
          <td>0.171009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.351412</td>
          <td>0.376887</td>
          <td>26.213896</td>
          <td>0.126372</td>
          <td>26.186759</td>
          <td>0.111029</td>
          <td>26.060393</td>
          <td>0.161745</td>
          <td>25.493824</td>
          <td>0.184934</td>
          <td>24.840093</td>
          <td>0.232285</td>
          <td>0.084949</td>
          <td>0.053185</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.339197</td>
          <td>1.376805</td>
          <td>27.121470</td>
          <td>0.268099</td>
          <td>26.567079</td>
          <td>0.152004</td>
          <td>26.154546</td>
          <td>0.172611</td>
          <td>25.909666</td>
          <td>0.257736</td>
          <td>25.298664</td>
          <td>0.332156</td>
          <td>0.024513</td>
          <td>0.018858</td>
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
          <td>27.987080</td>
          <td>1.288371</td>
          <td>26.437282</td>
          <td>0.188709</td>
          <td>26.063255</td>
          <td>0.125672</td>
          <td>25.114340</td>
          <td>0.090218</td>
          <td>24.881215</td>
          <td>0.137510</td>
          <td>23.964194</td>
          <td>0.139290</td>
          <td>0.220823</td>
          <td>0.217676</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.563844</td>
          <td>0.391960</td>
          <td>26.524819</td>
          <td>0.151620</td>
          <td>26.281995</td>
          <td>0.198951</td>
          <td>25.725929</td>
          <td>0.228803</td>
          <td>25.407437</td>
          <td>0.373441</td>
          <td>0.147223</td>
          <td>0.120762</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.220183</td>
          <td>2.017871</td>
          <td>29.154908</td>
          <td>1.075806</td>
          <td>30.549832</td>
          <td>2.018814</td>
          <td>25.957947</td>
          <td>0.134842</td>
          <td>24.979195</td>
          <td>0.108495</td>
          <td>24.196452</td>
          <td>0.122567</td>
          <td>0.103529</td>
          <td>0.058534</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.833539</td>
          <td>3.279392</td>
          <td>27.433968</td>
          <td>0.284553</td>
          <td>26.248485</td>
          <td>0.168667</td>
          <td>25.601222</td>
          <td>0.180879</td>
          <td>25.473707</td>
          <td>0.346911</td>
          <td>0.073886</td>
          <td>0.063951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.493477</td>
          <td>0.375974</td>
          <td>26.176146</td>
          <td>0.104889</td>
          <td>25.945220</td>
          <td>0.075403</td>
          <td>25.593972</td>
          <td>0.090145</td>
          <td>25.376137</td>
          <td>0.141181</td>
          <td>25.458211</td>
          <td>0.324896</td>
          <td>0.017565</td>
          <td>0.015458</td>
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
          <td>26.760009</td>
          <td>0.486114</td>
          <td>26.357678</td>
          <td>0.132578</td>
          <td>25.585422</td>
          <td>0.059919</td>
          <td>25.009412</td>
          <td>0.058982</td>
          <td>25.019012</td>
          <td>0.112978</td>
          <td>24.430256</td>
          <td>0.150862</td>
          <td>0.104253</td>
          <td>0.065162</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.848619</td>
          <td>0.498606</td>
          <td>26.714354</td>
          <td>0.170027</td>
          <td>26.233595</td>
          <td>0.099341</td>
          <td>25.109668</td>
          <td>0.060119</td>
          <td>24.769481</td>
          <td>0.084988</td>
          <td>24.455665</td>
          <td>0.144214</td>
          <td>0.048120</td>
          <td>0.039219</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.444971</td>
          <td>0.176387</td>
          <td>26.588935</td>
          <td>0.181948</td>
          <td>26.216001</td>
          <td>0.214189</td>
          <td>26.151805</td>
          <td>0.364053</td>
          <td>26.098433</td>
          <td>0.692811</td>
          <td>0.202512</td>
          <td>0.171009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.370592</td>
          <td>0.735694</td>
          <td>26.123064</td>
          <td>0.105461</td>
          <td>26.094274</td>
          <td>0.091278</td>
          <td>25.857016</td>
          <td>0.120705</td>
          <td>25.913407</td>
          <td>0.235434</td>
          <td>25.183865</td>
          <td>0.275668</td>
          <td>0.084949</td>
          <td>0.053185</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.163825</td>
          <td>1.166653</td>
          <td>27.331963</td>
          <td>0.279810</td>
          <td>26.577297</td>
          <td>0.131507</td>
          <td>26.322069</td>
          <td>0.169942</td>
          <td>26.053019</td>
          <td>0.250495</td>
          <td>25.252743</td>
          <td>0.276137</td>
          <td>0.024513</td>
          <td>0.018858</td>
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
