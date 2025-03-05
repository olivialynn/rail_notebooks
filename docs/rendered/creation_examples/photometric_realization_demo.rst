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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f722f244340>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>27.304518</td>
          <td>0.679776</td>
          <td>26.538676</td>
          <td>0.143180</td>
          <td>26.024884</td>
          <td>0.080597</td>
          <td>25.209905</td>
          <td>0.063956</td>
          <td>24.713627</td>
          <td>0.078860</td>
          <td>23.942322</td>
          <td>0.089842</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.806024</td>
          <td>0.404955</td>
          <td>26.753304</td>
          <td>0.152070</td>
          <td>26.087235</td>
          <td>0.138045</td>
          <td>25.805142</td>
          <td>0.202649</td>
          <td>25.669881</td>
          <td>0.382371</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.817017</td>
          <td>0.829345</td>
          <td>30.526903</td>
          <td>1.918005</td>
          <td>26.082511</td>
          <td>0.137484</td>
          <td>25.027880</td>
          <td>0.103951</td>
          <td>24.299494</td>
          <td>0.122765</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.535607</td>
          <td>0.688181</td>
          <td>27.755176</td>
          <td>0.348018</td>
          <td>26.119194</td>
          <td>0.141901</td>
          <td>25.272262</td>
          <td>0.128596</td>
          <td>24.839439</td>
          <td>0.194891</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.693212</td>
          <td>0.437262</td>
          <td>26.080377</td>
          <td>0.096149</td>
          <td>25.786771</td>
          <td>0.065292</td>
          <td>25.692317</td>
          <td>0.097894</td>
          <td>25.532546</td>
          <td>0.160874</td>
          <td>25.046059</td>
          <td>0.231603</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>27.479132</td>
          <td>0.764469</td>
          <td>26.448338</td>
          <td>0.132453</td>
          <td>25.468260</td>
          <td>0.049215</td>
          <td>25.086877</td>
          <td>0.057343</td>
          <td>24.845434</td>
          <td>0.088574</td>
          <td>24.815969</td>
          <td>0.191075</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.303976</td>
          <td>1.257430</td>
          <td>26.790501</td>
          <td>0.177538</td>
          <td>26.109519</td>
          <td>0.086840</td>
          <td>25.271018</td>
          <td>0.067515</td>
          <td>24.976018</td>
          <td>0.099336</td>
          <td>24.133977</td>
          <td>0.106281</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.337241</td>
          <td>0.331832</td>
          <td>26.623738</td>
          <td>0.154023</td>
          <td>26.437370</td>
          <td>0.115727</td>
          <td>26.263222</td>
          <td>0.160565</td>
          <td>25.940484</td>
          <td>0.226883</td>
          <td>25.527533</td>
          <td>0.342053</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.883713</td>
          <td>0.229750</td>
          <td>26.355131</td>
          <td>0.122185</td>
          <td>26.145839</td>
          <td>0.089660</td>
          <td>25.884762</td>
          <td>0.115824</td>
          <td>26.088589</td>
          <td>0.256367</td>
          <td>25.615493</td>
          <td>0.366519</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.087675</td>
          <td>0.227801</td>
          <td>26.618046</td>
          <td>0.135359</td>
          <td>26.029603</td>
          <td>0.131340</td>
          <td>26.121537</td>
          <td>0.263373</td>
          <td>26.387591</td>
          <td>0.648637</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.664210</td>
          <td>0.472989</td>
          <td>26.538287</td>
          <td>0.164541</td>
          <td>26.203983</td>
          <td>0.110851</td>
          <td>25.251599</td>
          <td>0.078646</td>
          <td>24.626967</td>
          <td>0.085923</td>
          <td>24.048882</td>
          <td>0.116479</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.278523</td>
          <td>1.206142</td>
          <td>26.664541</td>
          <td>0.164990</td>
          <td>26.349686</td>
          <td>0.203248</td>
          <td>26.006776</td>
          <td>0.278611</td>
          <td>24.977483</td>
          <td>0.256014</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.754598</td>
          <td>2.397554</td>
          <td>27.639867</td>
          <td>0.374539</td>
          <td>26.097809</td>
          <td>0.167924</td>
          <td>24.931115</td>
          <td>0.114671</td>
          <td>24.276217</td>
          <td>0.145033</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.636337</td>
          <td>0.856911</td>
          <td>26.762590</td>
          <td>0.191258</td>
          <td>26.491060</td>
          <td>0.244052</td>
          <td>26.027642</td>
          <td>0.301302</td>
          <td>25.742066</td>
          <td>0.494933</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.933663</td>
          <td>0.267200</td>
          <td>26.061441</td>
          <td>0.109082</td>
          <td>25.794774</td>
          <td>0.077418</td>
          <td>25.928061</td>
          <td>0.142020</td>
          <td>25.221058</td>
          <td>0.144230</td>
          <td>24.995286</td>
          <td>0.259803</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.226835</td>
          <td>1.309041</td>
          <td>26.537935</td>
          <td>0.167545</td>
          <td>25.353895</td>
          <td>0.053500</td>
          <td>25.091437</td>
          <td>0.069778</td>
          <td>24.748937</td>
          <td>0.097671</td>
          <td>24.881903</td>
          <td>0.241443</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.644859</td>
          <td>0.467463</td>
          <td>26.517152</td>
          <td>0.162194</td>
          <td>26.065312</td>
          <td>0.098603</td>
          <td>25.242915</td>
          <td>0.078386</td>
          <td>24.677813</td>
          <td>0.090229</td>
          <td>24.193372</td>
          <td>0.132590</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.479673</td>
          <td>0.414933</td>
          <td>26.494197</td>
          <td>0.160255</td>
          <td>26.504116</td>
          <td>0.145589</td>
          <td>26.285661</td>
          <td>0.195017</td>
          <td>26.074450</td>
          <td>0.297694</td>
          <td>25.734204</td>
          <td>0.469395</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.486200</td>
          <td>0.422213</td>
          <td>26.355385</td>
          <td>0.144653</td>
          <td>25.928700</td>
          <td>0.089877</td>
          <td>25.949112</td>
          <td>0.149238</td>
          <td>25.622185</td>
          <td>0.208927</td>
          <td>25.132443</td>
          <td>0.299071</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.216151</td>
          <td>0.705111</td>
          <td>26.958492</td>
          <td>0.236211</td>
          <td>26.478868</td>
          <td>0.142077</td>
          <td>26.387208</td>
          <td>0.211777</td>
          <td>26.042451</td>
          <td>0.289382</td>
          <td>25.231710</td>
          <td>0.317412</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.349280</td>
          <td>0.121580</td>
          <td>26.084540</td>
          <td>0.084961</td>
          <td>25.195462</td>
          <td>0.063151</td>
          <td>24.669229</td>
          <td>0.075838</td>
          <td>24.083609</td>
          <td>0.101714</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.324334</td>
          <td>0.689385</td>
          <td>27.944979</td>
          <td>0.450433</td>
          <td>26.713137</td>
          <td>0.147052</td>
          <td>26.411899</td>
          <td>0.182384</td>
          <td>25.943699</td>
          <td>0.227693</td>
          <td>25.691265</td>
          <td>0.389096</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.540846</td>
          <td>0.729892</td>
          <td>28.722017</td>
          <td>0.753995</td>
          <td>26.226706</td>
          <td>0.169114</td>
          <td>24.976158</td>
          <td>0.107782</td>
          <td>24.591130</td>
          <td>0.171389</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.267497</td>
          <td>0.289301</td>
          <td>26.102514</td>
          <td>0.175693</td>
          <td>25.557867</td>
          <td>0.204128</td>
          <td>25.125788</td>
          <td>0.306571</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.126719</td>
          <td>0.280589</td>
          <td>25.939246</td>
          <td>0.085048</td>
          <td>25.973460</td>
          <td>0.077130</td>
          <td>25.681672</td>
          <td>0.097129</td>
          <td>25.689285</td>
          <td>0.184055</td>
          <td>25.477056</td>
          <td>0.329090</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.898999</td>
          <td>0.534016</td>
          <td>26.364760</td>
          <td>0.131882</td>
          <td>25.489670</td>
          <td>0.054324</td>
          <td>25.109605</td>
          <td>0.063595</td>
          <td>24.757427</td>
          <td>0.088699</td>
          <td>24.515404</td>
          <td>0.160183</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.487786</td>
          <td>0.775488</td>
          <td>26.403602</td>
          <td>0.129239</td>
          <td>26.004481</td>
          <td>0.080482</td>
          <td>25.159185</td>
          <td>0.062221</td>
          <td>24.684967</td>
          <td>0.078174</td>
          <td>24.207272</td>
          <td>0.115242</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.198675</td>
          <td>0.306592</td>
          <td>26.525478</td>
          <td>0.147610</td>
          <td>26.470706</td>
          <td>0.125037</td>
          <td>26.527054</td>
          <td>0.210839</td>
          <td>25.621265</td>
          <td>0.181884</td>
          <td>25.417889</td>
          <td>0.328272</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.943117</td>
          <td>0.563362</td>
          <td>26.417927</td>
          <td>0.142465</td>
          <td>26.185858</td>
          <td>0.104136</td>
          <td>25.701106</td>
          <td>0.111151</td>
          <td>25.694214</td>
          <td>0.205973</td>
          <td>25.569335</td>
          <td>0.392699</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>31.895930</td>
          <td>4.488295</td>
          <td>26.687450</td>
          <td>0.168082</td>
          <td>26.866061</td>
          <td>0.173886</td>
          <td>26.412867</td>
          <td>0.189654</td>
          <td>26.064015</td>
          <td>0.260524</td>
          <td>25.919532</td>
          <td>0.478752</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
