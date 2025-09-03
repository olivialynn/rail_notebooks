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

    <pzflow.flow.Flow at 0x7f14604f2b30>



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
    0      23.994413  0.142202  0.108889  
    1      25.391064  0.157660  0.106435  
    2      24.304707  0.082120  0.044555  
    3      25.291103  0.032373  0.032275  
    4      25.096743  0.150633  0.100400  
    ...          ...       ...       ...  
    99995  24.737946  0.089506  0.050582  
    99996  24.224169  0.048541  0.047462  
    99997  25.613836  0.122342  0.075659  
    99998  25.274899  0.202655  0.138930  
    99999  25.699642  0.019651  0.016495  
    
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
          <td>27.780129</td>
          <td>0.926894</td>
          <td>26.733184</td>
          <td>0.169104</td>
          <td>26.008341</td>
          <td>0.079429</td>
          <td>25.191691</td>
          <td>0.062931</td>
          <td>24.711511</td>
          <td>0.078713</td>
          <td>24.088213</td>
          <td>0.102111</td>
          <td>0.142202</td>
          <td>0.108889</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.335255</td>
          <td>0.279112</td>
          <td>26.497601</td>
          <td>0.121950</td>
          <td>26.135427</td>
          <td>0.143898</td>
          <td>26.120414</td>
          <td>0.263131</td>
          <td>24.977812</td>
          <td>0.218837</td>
          <td>0.157660</td>
          <td>0.106435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.845658</td>
          <td>0.490136</td>
          <td>28.862146</td>
          <td>0.853676</td>
          <td>27.637925</td>
          <td>0.317123</td>
          <td>25.946476</td>
          <td>0.122209</td>
          <td>25.039813</td>
          <td>0.105041</td>
          <td>24.376709</td>
          <td>0.131261</td>
          <td>0.082120</td>
          <td>0.044555</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.303939</td>
          <td>0.679507</td>
          <td>28.640019</td>
          <td>0.738431</td>
          <td>27.033752</td>
          <td>0.193024</td>
          <td>26.445465</td>
          <td>0.187455</td>
          <td>25.408586</td>
          <td>0.144654</td>
          <td>24.993323</td>
          <td>0.221681</td>
          <td>0.032373</td>
          <td>0.032275</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.907746</td>
          <td>0.234360</td>
          <td>26.235925</td>
          <td>0.110152</td>
          <td>25.994353</td>
          <td>0.078454</td>
          <td>25.766600</td>
          <td>0.104475</td>
          <td>25.330569</td>
          <td>0.135247</td>
          <td>25.685932</td>
          <td>0.387158</td>
          <td>0.150633</td>
          <td>0.100400</td>
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
          <td>27.531553</td>
          <td>0.791264</td>
          <td>26.324588</td>
          <td>0.118988</td>
          <td>25.459689</td>
          <td>0.048842</td>
          <td>24.991731</td>
          <td>0.052698</td>
          <td>24.823937</td>
          <td>0.086914</td>
          <td>24.666662</td>
          <td>0.168366</td>
          <td>0.089506</td>
          <td>0.050582</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.831756</td>
          <td>0.956822</td>
          <td>26.766010</td>
          <td>0.173888</td>
          <td>26.031386</td>
          <td>0.081061</td>
          <td>25.194083</td>
          <td>0.063065</td>
          <td>24.799867</td>
          <td>0.085091</td>
          <td>24.116354</td>
          <td>0.104656</td>
          <td>0.048541</td>
          <td>0.047462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.849051</td>
          <td>0.491368</td>
          <td>26.560823</td>
          <td>0.145932</td>
          <td>26.181939</td>
          <td>0.092551</td>
          <td>26.230725</td>
          <td>0.156164</td>
          <td>26.147038</td>
          <td>0.268911</td>
          <td>25.813812</td>
          <td>0.427097</td>
          <td>0.122342</td>
          <td>0.075659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.753034</td>
          <td>0.457433</td>
          <td>26.264963</td>
          <td>0.112975</td>
          <td>26.091311</td>
          <td>0.085458</td>
          <td>25.866133</td>
          <td>0.113959</td>
          <td>25.515009</td>
          <td>0.158480</td>
          <td>25.728641</td>
          <td>0.400138</td>
          <td>0.202655</td>
          <td>0.138930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.907263</td>
          <td>0.234267</td>
          <td>27.444558</td>
          <td>0.304839</td>
          <td>26.526218</td>
          <td>0.125017</td>
          <td>26.243452</td>
          <td>0.157874</td>
          <td>25.689984</td>
          <td>0.183913</td>
          <td>24.951075</td>
          <td>0.214011</td>
          <td>0.019651</td>
          <td>0.016495</td>
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
          <td>28.689085</td>
          <td>1.673225</td>
          <td>26.602261</td>
          <td>0.181755</td>
          <td>26.015986</td>
          <td>0.099008</td>
          <td>25.171523</td>
          <td>0.077317</td>
          <td>24.856276</td>
          <td>0.110596</td>
          <td>23.959557</td>
          <td>0.113572</td>
          <td>0.142202</td>
          <td>0.108889</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.903523</td>
          <td>0.514728</td>
          <td>26.475784</td>
          <td>0.148447</td>
          <td>26.236086</td>
          <td>0.195513</td>
          <td>25.641849</td>
          <td>0.217720</td>
          <td>25.934316</td>
          <td>0.564734</td>
          <td>0.157660</td>
          <td>0.106435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.754677</td>
          <td>0.510475</td>
          <td>28.174832</td>
          <td>0.605993</td>
          <td>27.408218</td>
          <td>0.309759</td>
          <td>25.996812</td>
          <td>0.152847</td>
          <td>25.067620</td>
          <td>0.128117</td>
          <td>24.175772</td>
          <td>0.131966</td>
          <td>0.082120</td>
          <td>0.044555</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.948239</td>
          <td>1.847311</td>
          <td>28.619665</td>
          <td>0.812768</td>
          <td>27.398259</td>
          <td>0.304218</td>
          <td>26.271375</td>
          <td>0.190920</td>
          <td>25.512386</td>
          <td>0.185487</td>
          <td>25.764465</td>
          <td>0.476252</td>
          <td>0.032373</td>
          <td>0.032275</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.691375</td>
          <td>0.499332</td>
          <td>26.136362</td>
          <td>0.122005</td>
          <td>25.835506</td>
          <td>0.084564</td>
          <td>25.608966</td>
          <td>0.113595</td>
          <td>25.546066</td>
          <td>0.199976</td>
          <td>25.498320</td>
          <td>0.406564</td>
          <td>0.150633</td>
          <td>0.100400</td>
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
          <td>26.618244</td>
          <td>0.462260</td>
          <td>26.162874</td>
          <td>0.120997</td>
          <td>25.506789</td>
          <td>0.061057</td>
          <td>25.205122</td>
          <td>0.076879</td>
          <td>24.902429</td>
          <td>0.111317</td>
          <td>24.500151</td>
          <td>0.174792</td>
          <td>0.089506</td>
          <td>0.050582</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.848452</td>
          <td>0.215173</td>
          <td>26.054980</td>
          <td>0.098065</td>
          <td>25.195390</td>
          <td>0.075446</td>
          <td>25.056736</td>
          <td>0.126068</td>
          <td>24.554933</td>
          <td>0.181330</td>
          <td>0.048541</td>
          <td>0.047462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.270837</td>
          <td>0.741773</td>
          <td>26.927867</td>
          <td>0.235014</td>
          <td>26.678121</td>
          <td>0.172427</td>
          <td>26.691531</td>
          <td>0.278463</td>
          <td>25.821097</td>
          <td>0.246969</td>
          <td>26.229990</td>
          <td>0.681443</td>
          <td>0.122342</td>
          <td>0.075659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.860451</td>
          <td>0.268994</td>
          <td>26.348627</td>
          <td>0.151835</td>
          <td>26.018164</td>
          <td>0.103299</td>
          <td>25.773099</td>
          <td>0.136378</td>
          <td>25.752978</td>
          <td>0.246624</td>
          <td>24.840296</td>
          <td>0.249808</td>
          <td>0.202655</td>
          <td>0.138930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.068966</td>
          <td>0.633917</td>
          <td>26.915901</td>
          <td>0.226297</td>
          <td>26.599784</td>
          <td>0.156245</td>
          <td>26.360305</td>
          <td>0.205245</td>
          <td>26.016937</td>
          <td>0.281145</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.019651</td>
          <td>0.016495</td>
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
          <td>27.145821</td>
          <td>0.674844</td>
          <td>26.932932</td>
          <td>0.232569</td>
          <td>25.896628</td>
          <td>0.086077</td>
          <td>25.165066</td>
          <td>0.074117</td>
          <td>24.691261</td>
          <td>0.092432</td>
          <td>23.938262</td>
          <td>0.107560</td>
          <td>0.142202</td>
          <td>0.108889</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.397847</td>
          <td>0.343521</td>
          <td>26.439597</td>
          <td>0.140476</td>
          <td>26.328422</td>
          <td>0.206190</td>
          <td>25.597659</td>
          <td>0.204985</td>
          <td>25.154941</td>
          <td>0.304939</td>
          <td>0.157660</td>
          <td>0.106435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.734500</td>
          <td>0.745037</td>
          <td>26.249190</td>
          <td>0.167786</td>
          <td>25.104957</td>
          <td>0.117426</td>
          <td>24.272431</td>
          <td>0.126872</td>
          <td>0.082120</td>
          <td>0.044555</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.308095</td>
          <td>2.042616</td>
          <td>28.927746</td>
          <td>0.897624</td>
          <td>27.094400</td>
          <td>0.205850</td>
          <td>25.950045</td>
          <td>0.124382</td>
          <td>25.504007</td>
          <td>0.159147</td>
          <td>25.444582</td>
          <td>0.324516</td>
          <td>0.032373</td>
          <td>0.032275</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.997412</td>
          <td>0.609349</td>
          <td>26.166000</td>
          <td>0.121374</td>
          <td>25.793006</td>
          <td>0.078660</td>
          <td>25.420888</td>
          <td>0.092967</td>
          <td>25.616851</td>
          <td>0.205295</td>
          <td>25.484457</td>
          <td>0.389849</td>
          <td>0.150633</td>
          <td>0.100400</td>
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
          <td>26.825681</td>
          <td>0.502296</td>
          <td>26.644588</td>
          <td>0.165839</td>
          <td>25.441625</td>
          <td>0.051382</td>
          <td>25.068978</td>
          <td>0.060515</td>
          <td>24.806857</td>
          <td>0.091453</td>
          <td>24.828128</td>
          <td>0.206055</td>
          <td>0.089506</td>
          <td>0.050582</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.720965</td>
          <td>0.454712</td>
          <td>26.730346</td>
          <td>0.173064</td>
          <td>26.012613</td>
          <td>0.082199</td>
          <td>25.230271</td>
          <td>0.067249</td>
          <td>24.895628</td>
          <td>0.095420</td>
          <td>24.525798</td>
          <td>0.153922</td>
          <td>0.048541</td>
          <td>0.047462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.222441</td>
          <td>0.687764</td>
          <td>26.618875</td>
          <td>0.170125</td>
          <td>26.360535</td>
          <td>0.122002</td>
          <td>26.630852</td>
          <td>0.246615</td>
          <td>25.679544</td>
          <td>0.204652</td>
          <td>26.601944</td>
          <td>0.823462</td>
          <td>0.122342</td>
          <td>0.075659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.080929</td>
          <td>0.329867</td>
          <td>26.176506</td>
          <td>0.135296</td>
          <td>25.897986</td>
          <td>0.096356</td>
          <td>25.646465</td>
          <td>0.126717</td>
          <td>25.342349</td>
          <td>0.181064</td>
          <td>25.493171</td>
          <td>0.433012</td>
          <td>0.202655</td>
          <td>0.138930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.165119</td>
          <td>1.166464</td>
          <td>26.737737</td>
          <td>0.170396</td>
          <td>26.519526</td>
          <td>0.124841</td>
          <td>26.493696</td>
          <td>0.196111</td>
          <td>26.473998</td>
          <td>0.350865</td>
          <td>25.584447</td>
          <td>0.359199</td>
          <td>0.019651</td>
          <td>0.016495</td>
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
