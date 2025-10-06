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

    <pzflow.flow.Flow at 0x7fc6e4744700>



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
    0      23.994413  0.053496  0.036203  
    1      25.391064  0.067229  0.049642  
    2      24.304707  0.193339  0.190474  
    3      25.291103  0.063777  0.054517  
    4      25.096743  0.053887  0.045665  
    ...          ...       ...       ...  
    99995  24.737946  0.068395  0.063274  
    99996  24.224169  0.074712  0.038124  
    99997  25.613836  0.153598  0.114692  
    99998  25.274899  0.032442  0.026820  
    99999  25.699642  0.184466  0.174168  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.525030</td>
          <td>0.141509</td>
          <td>26.080412</td>
          <td>0.084642</td>
          <td>25.250912</td>
          <td>0.066323</td>
          <td>24.742829</td>
          <td>0.080919</td>
          <td>24.206049</td>
          <td>0.113181</td>
          <td>0.053496</td>
          <td>0.036203</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.637168</td>
          <td>1.496485</td>
          <td>27.757875</td>
          <td>0.390205</td>
          <td>26.589565</td>
          <td>0.132068</td>
          <td>26.479513</td>
          <td>0.192917</td>
          <td>25.807686</td>
          <td>0.203082</td>
          <td>25.313663</td>
          <td>0.288322</td>
          <td>0.067229</td>
          <td>0.049642</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.412138</td>
          <td>0.731145</td>
          <td>28.537331</td>
          <td>0.688990</td>
          <td>28.348912</td>
          <td>0.545492</td>
          <td>25.930362</td>
          <td>0.120510</td>
          <td>25.028833</td>
          <td>0.104037</td>
          <td>24.178953</td>
          <td>0.110538</td>
          <td>0.193339</td>
          <td>0.190474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.228429</td>
          <td>3.599757</td>
          <td>27.245975</td>
          <td>0.230488</td>
          <td>26.433909</td>
          <td>0.185634</td>
          <td>25.581004</td>
          <td>0.167663</td>
          <td>25.538537</td>
          <td>0.345036</td>
          <td>0.063777</td>
          <td>0.054517</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.748221</td>
          <td>0.205259</td>
          <td>26.067568</td>
          <td>0.095076</td>
          <td>25.871704</td>
          <td>0.070393</td>
          <td>25.751221</td>
          <td>0.103078</td>
          <td>25.284228</td>
          <td>0.129935</td>
          <td>25.185875</td>
          <td>0.259862</td>
          <td>0.053887</td>
          <td>0.045665</td>
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
          <td>26.730131</td>
          <td>0.449624</td>
          <td>26.326425</td>
          <td>0.119178</td>
          <td>25.438519</td>
          <td>0.047933</td>
          <td>24.979526</td>
          <td>0.052130</td>
          <td>24.692792</td>
          <td>0.077423</td>
          <td>24.601614</td>
          <td>0.159276</td>
          <td>0.068395</td>
          <td>0.063274</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.974556</td>
          <td>0.538694</td>
          <td>26.600606</td>
          <td>0.151001</td>
          <td>26.001807</td>
          <td>0.078972</td>
          <td>25.141962</td>
          <td>0.060216</td>
          <td>24.796031</td>
          <td>0.084804</td>
          <td>24.191994</td>
          <td>0.111803</td>
          <td>0.074712</td>
          <td>0.038124</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.957722</td>
          <td>0.532147</td>
          <td>26.678182</td>
          <td>0.161361</td>
          <td>26.324463</td>
          <td>0.104868</td>
          <td>26.139399</td>
          <td>0.144391</td>
          <td>26.131580</td>
          <td>0.265542</td>
          <td>26.193493</td>
          <td>0.565624</td>
          <td>0.153598</td>
          <td>0.114692</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.303065</td>
          <td>0.322954</td>
          <td>26.443033</td>
          <td>0.131848</td>
          <td>26.277523</td>
          <td>0.100647</td>
          <td>26.008338</td>
          <td>0.128945</td>
          <td>25.545228</td>
          <td>0.162625</td>
          <td>26.500551</td>
          <td>0.700887</td>
          <td>0.032442</td>
          <td>0.026820</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.756613</td>
          <td>0.458663</td>
          <td>26.582137</td>
          <td>0.148628</td>
          <td>26.503965</td>
          <td>0.122626</td>
          <td>26.109833</td>
          <td>0.140761</td>
          <td>25.822572</td>
          <td>0.205632</td>
          <td>26.088571</td>
          <td>0.524249</td>
          <td>0.184466</td>
          <td>0.174168</td>
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
          <td>26.729788</td>
          <td>0.498801</td>
          <td>26.789027</td>
          <td>0.204619</td>
          <td>25.997474</td>
          <td>0.093171</td>
          <td>25.262773</td>
          <td>0.080008</td>
          <td>24.696517</td>
          <td>0.091986</td>
          <td>23.908518</td>
          <td>0.103797</td>
          <td>0.053496</td>
          <td>0.036203</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.698568</td>
          <td>3.432905</td>
          <td>27.607109</td>
          <td>0.397447</td>
          <td>26.440811</td>
          <td>0.137720</td>
          <td>26.534466</td>
          <td>0.239721</td>
          <td>25.798780</td>
          <td>0.237505</td>
          <td>25.152192</td>
          <td>0.298287</td>
          <td>0.067229</td>
          <td>0.049642</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.747407</td>
          <td>1.045800</td>
          <td>29.083855</td>
          <td>1.153500</td>
          <td>28.203429</td>
          <td>0.613632</td>
          <td>26.397558</td>
          <td>0.235813</td>
          <td>25.161707</td>
          <td>0.152781</td>
          <td>24.410546</td>
          <td>0.177763</td>
          <td>0.193339</td>
          <td>0.190474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>32.389697</td>
          <td>5.087176</td>
          <td>29.472249</td>
          <td>1.348250</td>
          <td>27.222949</td>
          <td>0.266000</td>
          <td>26.893987</td>
          <td>0.320972</td>
          <td>25.541850</td>
          <td>0.191678</td>
          <td>25.150572</td>
          <td>0.297927</td>
          <td>0.063777</td>
          <td>0.054517</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.433148</td>
          <td>0.399221</td>
          <td>26.092325</td>
          <td>0.112870</td>
          <td>26.006507</td>
          <td>0.094042</td>
          <td>25.753162</td>
          <td>0.123099</td>
          <td>25.471216</td>
          <td>0.179980</td>
          <td>25.436518</td>
          <td>0.372491</td>
          <td>0.053887</td>
          <td>0.045665</td>
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
          <td>26.953289</td>
          <td>0.589183</td>
          <td>26.364829</td>
          <td>0.143666</td>
          <td>25.451276</td>
          <td>0.057946</td>
          <td>25.038995</td>
          <td>0.066165</td>
          <td>24.717862</td>
          <td>0.094434</td>
          <td>24.522078</td>
          <td>0.177541</td>
          <td>0.068395</td>
          <td>0.063274</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.884446</td>
          <td>0.559943</td>
          <td>26.488869</td>
          <td>0.159385</td>
          <td>25.995696</td>
          <td>0.093469</td>
          <td>25.110657</td>
          <td>0.070294</td>
          <td>25.091856</td>
          <td>0.130474</td>
          <td>24.313200</td>
          <td>0.148139</td>
          <td>0.074712</td>
          <td>0.038124</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.915230</td>
          <td>0.589475</td>
          <td>26.837954</td>
          <td>0.222829</td>
          <td>26.468740</td>
          <td>0.147699</td>
          <td>26.411797</td>
          <td>0.226673</td>
          <td>25.650578</td>
          <td>0.219522</td>
          <td>25.651129</td>
          <td>0.459080</td>
          <td>0.153598</td>
          <td>0.114692</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.995124</td>
          <td>0.281391</td>
          <td>26.301864</td>
          <td>0.134696</td>
          <td>26.204201</td>
          <td>0.111200</td>
          <td>26.301488</td>
          <td>0.195720</td>
          <td>25.449496</td>
          <td>0.175775</td>
          <td>25.730056</td>
          <td>0.463945</td>
          <td>0.032442</td>
          <td>0.026820</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.424504</td>
          <td>0.421661</td>
          <td>26.826303</td>
          <td>0.228472</td>
          <td>26.727016</td>
          <td>0.191313</td>
          <td>26.201660</td>
          <td>0.197802</td>
          <td>26.008873</td>
          <td>0.305469</td>
          <td>26.450670</td>
          <td>0.829612</td>
          <td>0.184466</td>
          <td>0.174168</td>
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
          <td>27.099606</td>
          <td>0.598464</td>
          <td>27.067172</td>
          <td>0.229067</td>
          <td>26.095340</td>
          <td>0.088156</td>
          <td>25.218805</td>
          <td>0.066365</td>
          <td>24.560380</td>
          <td>0.070800</td>
          <td>23.920969</td>
          <td>0.090725</td>
          <td>0.053496</td>
          <td>0.036203</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.678491</td>
          <td>0.444371</td>
          <td>27.663482</td>
          <td>0.375467</td>
          <td>26.617866</td>
          <td>0.141465</td>
          <td>26.162393</td>
          <td>0.154230</td>
          <td>25.833924</td>
          <td>0.216679</td>
          <td>25.288339</td>
          <td>0.294887</td>
          <td>0.067229</td>
          <td>0.049642</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.002088</td>
          <td>0.676939</td>
          <td>28.166025</td>
          <td>0.679513</td>
          <td>33.215619</td>
          <td>4.805522</td>
          <td>26.192762</td>
          <td>0.212778</td>
          <td>24.935555</td>
          <td>0.134623</td>
          <td>24.316852</td>
          <td>0.175814</td>
          <td>0.193339</td>
          <td>0.190474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.065210</td>
          <td>0.590115</td>
          <td>28.793278</td>
          <td>0.840843</td>
          <td>27.578424</td>
          <td>0.315183</td>
          <td>26.317173</td>
          <td>0.176076</td>
          <td>25.404788</td>
          <td>0.150745</td>
          <td>25.074117</td>
          <td>0.247790</td>
          <td>0.063777</td>
          <td>0.054517</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.008531</td>
          <td>0.260101</td>
          <td>26.068534</td>
          <td>0.097905</td>
          <td>26.114950</td>
          <td>0.090167</td>
          <td>25.850118</td>
          <td>0.116273</td>
          <td>25.421919</td>
          <td>0.151075</td>
          <td>25.701464</td>
          <td>0.403686</td>
          <td>0.053887</td>
          <td>0.045665</td>
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
          <td>27.064933</td>
          <td>0.593429</td>
          <td>26.401325</td>
          <td>0.133410</td>
          <td>25.509286</td>
          <td>0.054001</td>
          <td>25.180743</td>
          <td>0.066104</td>
          <td>24.898799</td>
          <td>0.098132</td>
          <td>25.142200</td>
          <td>0.264570</td>
          <td>0.068395</td>
          <td>0.063274</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.882454</td>
          <td>0.517120</td>
          <td>26.550092</td>
          <td>0.150247</td>
          <td>25.888980</td>
          <td>0.074786</td>
          <td>25.209838</td>
          <td>0.067064</td>
          <td>25.066553</td>
          <td>0.112426</td>
          <td>24.382990</td>
          <td>0.138160</td>
          <td>0.074712</td>
          <td>0.038124</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.904113</td>
          <td>0.577146</td>
          <td>26.622347</td>
          <td>0.182570</td>
          <td>26.358052</td>
          <td>0.131459</td>
          <td>26.280038</td>
          <td>0.198805</td>
          <td>26.172159</td>
          <td>0.329066</td>
          <td>29.260704</td>
          <td>2.960441</td>
          <td>0.153598</td>
          <td>0.114692</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.415472</td>
          <td>0.355558</td>
          <td>26.278801</td>
          <td>0.115520</td>
          <td>26.127779</td>
          <td>0.089305</td>
          <td>25.911904</td>
          <td>0.120066</td>
          <td>26.370494</td>
          <td>0.325454</td>
          <td>25.019228</td>
          <td>0.229163</td>
          <td>0.032442</td>
          <td>0.026820</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.687594</td>
          <td>1.741701</td>
          <td>26.646674</td>
          <td>0.205538</td>
          <td>26.614543</td>
          <td>0.182569</td>
          <td>26.662648</td>
          <td>0.303301</td>
          <td>26.005641</td>
          <td>0.318890</td>
          <td>29.797319</td>
          <td>3.593963</td>
          <td>0.184466</td>
          <td>0.174168</td>
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
