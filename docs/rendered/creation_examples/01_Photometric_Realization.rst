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

    <pzflow.flow.Flow at 0x7f837a136ad0>



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
    0      23.994413  0.058176  0.047909  
    1      25.391064  0.091919  0.074797  
    2      24.304707  0.040825  0.030919  
    3      25.291103  0.023358  0.013954  
    4      25.096743  0.053400  0.045888  
    ...          ...       ...       ...  
    99995  24.737946  0.045902  0.044845  
    99996  24.224169  0.054317  0.046750  
    99997  25.613836  0.079778  0.078431  
    99998  25.274899  0.261388  0.184316  
    99999  25.699642  0.136710  0.105493  
    
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
          <td>27.310144</td>
          <td>0.682395</td>
          <td>26.726429</td>
          <td>0.168135</td>
          <td>26.081554</td>
          <td>0.084727</td>
          <td>25.261888</td>
          <td>0.066971</td>
          <td>24.780183</td>
          <td>0.083628</td>
          <td>23.908867</td>
          <td>0.087236</td>
          <td>0.058176</td>
          <td>0.047909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.529813</td>
          <td>0.326312</td>
          <td>26.548818</td>
          <td>0.127490</td>
          <td>26.433030</td>
          <td>0.185496</td>
          <td>26.019576</td>
          <td>0.242227</td>
          <td>25.260886</td>
          <td>0.276252</td>
          <td>0.091919</td>
          <td>0.074797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.113209</td>
          <td>0.594994</td>
          <td>28.723902</td>
          <td>0.780619</td>
          <td>27.810021</td>
          <td>0.363327</td>
          <td>25.890763</td>
          <td>0.116430</td>
          <td>25.068197</td>
          <td>0.107679</td>
          <td>24.263580</td>
          <td>0.118994</td>
          <td>0.040825</td>
          <td>0.030919</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.146363</td>
          <td>1.151860</td>
          <td>28.037013</td>
          <td>0.482212</td>
          <td>27.275572</td>
          <td>0.236205</td>
          <td>26.482559</td>
          <td>0.193413</td>
          <td>25.212651</td>
          <td>0.122117</td>
          <td>25.662592</td>
          <td>0.380214</td>
          <td>0.023358</td>
          <td>0.013954</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.721203</td>
          <td>0.446609</td>
          <td>26.123479</td>
          <td>0.099848</td>
          <td>26.000470</td>
          <td>0.078879</td>
          <td>25.539757</td>
          <td>0.085609</td>
          <td>25.628967</td>
          <td>0.174644</td>
          <td>25.200923</td>
          <td>0.263079</td>
          <td>0.053400</td>
          <td>0.045888</td>
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
          <td>26.565201</td>
          <td>0.396533</td>
          <td>26.416221</td>
          <td>0.128826</td>
          <td>25.358128</td>
          <td>0.044631</td>
          <td>25.049124</td>
          <td>0.055453</td>
          <td>24.797516</td>
          <td>0.084915</td>
          <td>24.417118</td>
          <td>0.135927</td>
          <td>0.045902</td>
          <td>0.044845</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.596307</td>
          <td>0.825235</td>
          <td>26.704199</td>
          <td>0.164982</td>
          <td>26.010768</td>
          <td>0.079600</td>
          <td>25.183158</td>
          <td>0.062457</td>
          <td>24.911686</td>
          <td>0.093885</td>
          <td>24.190384</td>
          <td>0.111646</td>
          <td>0.054317</td>
          <td>0.046750</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.440101</td>
          <td>0.359824</td>
          <td>26.867536</td>
          <td>0.189486</td>
          <td>26.460076</td>
          <td>0.118036</td>
          <td>26.092793</td>
          <td>0.138709</td>
          <td>26.047409</td>
          <td>0.247844</td>
          <td>25.881838</td>
          <td>0.449681</td>
          <td>0.079778</td>
          <td>0.078431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.785193</td>
          <td>0.211695</td>
          <td>26.275860</td>
          <td>0.114052</td>
          <td>26.169639</td>
          <td>0.091556</td>
          <td>25.831042</td>
          <td>0.110525</td>
          <td>26.060849</td>
          <td>0.250597</td>
          <td>24.744747</td>
          <td>0.179912</td>
          <td>0.261388</td>
          <td>0.184316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.032087</td>
          <td>0.217514</td>
          <td>26.372456</td>
          <td>0.109359</td>
          <td>26.220220</td>
          <td>0.154765</td>
          <td>26.071113</td>
          <td>0.252718</td>
          <td>25.261611</td>
          <td>0.276415</td>
          <td>0.136710</td>
          <td>0.105493</td>
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
          <td>26.593310</td>
          <td>0.173868</td>
          <td>26.014345</td>
          <td>0.094801</td>
          <td>25.186218</td>
          <td>0.074977</td>
          <td>24.723900</td>
          <td>0.094463</td>
          <td>24.048866</td>
          <td>0.117616</td>
          <td>0.058176</td>
          <td>0.047909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.584645</td>
          <td>0.394230</td>
          <td>26.593124</td>
          <td>0.158755</td>
          <td>26.118057</td>
          <td>0.171015</td>
          <td>25.815895</td>
          <td>0.243515</td>
          <td>24.969581</td>
          <td>0.260048</td>
          <td>0.091919</td>
          <td>0.074797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.892328</td>
          <td>0.447489</td>
          <td>26.261760</td>
          <td>0.189548</td>
          <td>24.879610</td>
          <td>0.107706</td>
          <td>24.105023</td>
          <td>0.122853</td>
          <td>0.040825</td>
          <td>0.030919</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.619521</td>
          <td>1.584990</td>
          <td>28.335070</td>
          <td>0.671080</td>
          <td>26.934981</td>
          <td>0.207560</td>
          <td>26.403716</td>
          <td>0.212864</td>
          <td>25.469576</td>
          <td>0.178492</td>
          <td>25.687787</td>
          <td>0.448748</td>
          <td>0.023358</td>
          <td>0.013954</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.789814</td>
          <td>0.238856</td>
          <td>25.966233</td>
          <td>0.101108</td>
          <td>25.860477</td>
          <td>0.082702</td>
          <td>25.610461</td>
          <td>0.108714</td>
          <td>25.169626</td>
          <td>0.139070</td>
          <td>24.923786</td>
          <td>0.246892</td>
          <td>0.053400</td>
          <td>0.045888</td>
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
          <td>27.487608</td>
          <td>0.841694</td>
          <td>26.594604</td>
          <td>0.173663</td>
          <td>25.549155</td>
          <td>0.062715</td>
          <td>25.088652</td>
          <td>0.068592</td>
          <td>24.708565</td>
          <td>0.092957</td>
          <td>24.916808</td>
          <td>0.245146</td>
          <td>0.045902</td>
          <td>0.044845</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.505706</td>
          <td>2.326029</td>
          <td>26.662917</td>
          <td>0.184279</td>
          <td>26.003953</td>
          <td>0.093855</td>
          <td>25.139675</td>
          <td>0.071887</td>
          <td>24.836672</td>
          <td>0.104177</td>
          <td>24.348028</td>
          <td>0.152155</td>
          <td>0.054317</td>
          <td>0.046750</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.793091</td>
          <td>0.527120</td>
          <td>27.111026</td>
          <td>0.270146</td>
          <td>26.739378</td>
          <td>0.179372</td>
          <td>26.181662</td>
          <td>0.180057</td>
          <td>26.452440</td>
          <td>0.403783</td>
          <td>25.500143</td>
          <td>0.395761</td>
          <td>0.079778</td>
          <td>0.078431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.112126</td>
          <td>0.342506</td>
          <td>26.010779</td>
          <td>0.119205</td>
          <td>26.120819</td>
          <td>0.119254</td>
          <td>25.930129</td>
          <td>0.164814</td>
          <td>25.803684</td>
          <td>0.270453</td>
          <td>25.196562</td>
          <td>0.350160</td>
          <td>0.261388</td>
          <td>0.184316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.863523</td>
          <td>0.225572</td>
          <td>26.482495</td>
          <td>0.147937</td>
          <td>26.247038</td>
          <td>0.195498</td>
          <td>25.462775</td>
          <td>0.185663</td>
          <td>25.384650</td>
          <td>0.370901</td>
          <td>0.136710</td>
          <td>0.105493</td>
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
          <td>27.265909</td>
          <td>0.675566</td>
          <td>26.676294</td>
          <td>0.166235</td>
          <td>25.844365</td>
          <td>0.071329</td>
          <td>25.203362</td>
          <td>0.066133</td>
          <td>24.680825</td>
          <td>0.079518</td>
          <td>23.862838</td>
          <td>0.087069</td>
          <td>0.058176</td>
          <td>0.047909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.321926</td>
          <td>0.346547</td>
          <td>27.318106</td>
          <td>0.295222</td>
          <td>26.431548</td>
          <td>0.125463</td>
          <td>26.252601</td>
          <td>0.173752</td>
          <td>25.912680</td>
          <td>0.240575</td>
          <td>25.710776</td>
          <td>0.426831</td>
          <td>0.091919</td>
          <td>0.074797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.707740</td>
          <td>0.446737</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.715632</td>
          <td>0.714897</td>
          <td>26.031994</td>
          <td>0.134010</td>
          <td>25.056027</td>
          <td>0.108408</td>
          <td>24.294930</td>
          <td>0.124494</td>
          <td>0.040825</td>
          <td>0.030919</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.694151</td>
          <td>1.542364</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.272605</td>
          <td>0.236737</td>
          <td>26.202347</td>
          <td>0.153196</td>
          <td>25.677735</td>
          <td>0.182894</td>
          <td>25.990332</td>
          <td>0.489824</td>
          <td>0.023358</td>
          <td>0.013954</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.143190</td>
          <td>0.290102</td>
          <td>26.146150</td>
          <td>0.104764</td>
          <td>25.938623</td>
          <td>0.077173</td>
          <td>25.633007</td>
          <td>0.096154</td>
          <td>25.445064</td>
          <td>0.154073</td>
          <td>25.246074</td>
          <td>0.281597</td>
          <td>0.053400</td>
          <td>0.045888</td>
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
          <td>27.210047</td>
          <td>0.646499</td>
          <td>26.493952</td>
          <td>0.141008</td>
          <td>25.499624</td>
          <td>0.052020</td>
          <td>25.089380</td>
          <td>0.059155</td>
          <td>24.755253</td>
          <td>0.084074</td>
          <td>25.193027</td>
          <td>0.268336</td>
          <td>0.045902</td>
          <td>0.044845</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.005061</td>
          <td>1.078385</td>
          <td>26.662362</td>
          <td>0.163798</td>
          <td>25.994610</td>
          <td>0.081175</td>
          <td>25.100310</td>
          <td>0.060142</td>
          <td>24.858012</td>
          <td>0.092628</td>
          <td>24.125135</td>
          <td>0.109195</td>
          <td>0.054317</td>
          <td>0.046750</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.711722</td>
          <td>0.923863</td>
          <td>26.745154</td>
          <td>0.182526</td>
          <td>26.380488</td>
          <td>0.119023</td>
          <td>26.333427</td>
          <td>0.184498</td>
          <td>25.514609</td>
          <td>0.170923</td>
          <td>25.206518</td>
          <td>0.284883</td>
          <td>0.079778</td>
          <td>0.078431</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.708349</td>
          <td>0.580782</td>
          <td>26.396863</td>
          <td>0.183750</td>
          <td>25.943923</td>
          <td>0.114166</td>
          <td>25.783084</td>
          <td>0.162353</td>
          <td>25.656221</td>
          <td>0.265876</td>
          <td>24.657114</td>
          <td>0.251642</td>
          <td>0.261388</td>
          <td>0.184316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.548674</td>
          <td>1.534385</td>
          <td>26.699002</td>
          <td>0.189571</td>
          <td>26.556934</td>
          <td>0.151342</td>
          <td>26.216138</td>
          <td>0.182666</td>
          <td>25.819894</td>
          <td>0.240396</td>
          <td>24.820279</td>
          <td>0.226134</td>
          <td>0.136710</td>
          <td>0.105493</td>
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
