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

    <pzflow.flow.Flow at 0x7fae363d49d0>



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
    0      23.994413  0.184964  0.125776  
    1      25.391064  0.020030  0.013760  
    2      24.304707  0.157714  0.144095  
    3      25.291103  0.073655  0.051041  
    4      25.096743  0.061370  0.048975  
    ...          ...       ...       ...  
    99995  24.737946  0.133861  0.107340  
    99996  24.224169  0.158380  0.121546  
    99997  25.613836  0.148138  0.135931  
    99998  25.274899  0.042812  0.029060  
    99999  25.699642  0.026892  0.014157  
    
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
          <td>27.629713</td>
          <td>0.843136</td>
          <td>26.935188</td>
          <td>0.200584</td>
          <td>26.085992</td>
          <td>0.085059</td>
          <td>25.073534</td>
          <td>0.056668</td>
          <td>24.674900</td>
          <td>0.076209</td>
          <td>24.040423</td>
          <td>0.097924</td>
          <td>0.184964</td>
          <td>0.125776</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.002702</td>
          <td>1.060139</td>
          <td>27.544762</td>
          <td>0.330208</td>
          <td>26.311063</td>
          <td>0.103646</td>
          <td>26.397176</td>
          <td>0.179952</td>
          <td>25.649607</td>
          <td>0.177730</td>
          <td>26.057763</td>
          <td>0.512558</td>
          <td>0.020030</td>
          <td>0.013760</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.348038</td>
          <td>1.888877</td>
          <td>28.359689</td>
          <td>0.549759</td>
          <td>25.999877</td>
          <td>0.128003</td>
          <td>24.845088</td>
          <td>0.088547</td>
          <td>24.225948</td>
          <td>0.115160</td>
          <td>0.157714</td>
          <td>0.144095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.368370</td>
          <td>0.709933</td>
          <td>27.624521</td>
          <td>0.351675</td>
          <td>27.130494</td>
          <td>0.209355</td>
          <td>26.243704</td>
          <td>0.157908</td>
          <td>25.558348</td>
          <td>0.164456</td>
          <td>25.336721</td>
          <td>0.293739</td>
          <td>0.073655</td>
          <td>0.051041</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.602113</td>
          <td>0.828328</td>
          <td>26.096290</td>
          <td>0.097499</td>
          <td>26.024070</td>
          <td>0.080540</td>
          <td>25.597210</td>
          <td>0.090050</td>
          <td>25.292850</td>
          <td>0.130908</td>
          <td>24.884436</td>
          <td>0.202402</td>
          <td>0.061370</td>
          <td>0.048975</td>
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
          <td>26.751819</td>
          <td>0.457017</td>
          <td>26.364240</td>
          <td>0.123154</td>
          <td>25.401469</td>
          <td>0.046382</td>
          <td>25.012315</td>
          <td>0.053670</td>
          <td>24.901374</td>
          <td>0.093039</td>
          <td>24.692805</td>
          <td>0.172153</td>
          <td>0.133861</td>
          <td>0.107340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.646357</td>
          <td>0.852151</td>
          <td>26.466774</td>
          <td>0.134579</td>
          <td>26.023187</td>
          <td>0.080477</td>
          <td>25.179463</td>
          <td>0.062252</td>
          <td>24.943642</td>
          <td>0.096556</td>
          <td>24.132144</td>
          <td>0.106111</td>
          <td>0.158380</td>
          <td>0.121546</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.866250</td>
          <td>0.497653</td>
          <td>26.680823</td>
          <td>0.161725</td>
          <td>26.289291</td>
          <td>0.101690</td>
          <td>26.133079</td>
          <td>0.143608</td>
          <td>25.740567</td>
          <td>0.191938</td>
          <td>24.987957</td>
          <td>0.220693</td>
          <td>0.148138</td>
          <td>0.135931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.981123</td>
          <td>0.248955</td>
          <td>26.218323</td>
          <td>0.108475</td>
          <td>26.053553</td>
          <td>0.082661</td>
          <td>25.921633</td>
          <td>0.119600</td>
          <td>25.577732</td>
          <td>0.167196</td>
          <td>25.388408</td>
          <td>0.306204</td>
          <td>0.042812</td>
          <td>0.029060</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.796884</td>
          <td>0.178501</td>
          <td>26.722154</td>
          <td>0.148059</td>
          <td>26.213057</td>
          <td>0.153818</td>
          <td>25.710364</td>
          <td>0.187109</td>
          <td>25.873170</td>
          <td>0.446750</td>
          <td>0.026892</td>
          <td>0.014157</td>
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
          <td>28.934766</td>
          <td>1.889957</td>
          <td>26.746239</td>
          <td>0.209891</td>
          <td>26.079880</td>
          <td>0.107461</td>
          <td>25.181086</td>
          <td>0.080113</td>
          <td>24.911989</td>
          <td>0.119138</td>
          <td>23.883206</td>
          <td>0.109120</td>
          <td>0.184964</td>
          <td>0.125776</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.586334</td>
          <td>0.387806</td>
          <td>26.473354</td>
          <td>0.140153</td>
          <td>26.440600</td>
          <td>0.219458</td>
          <td>26.318665</td>
          <td>0.357619</td>
          <td>25.605634</td>
          <td>0.421545</td>
          <td>0.020030</td>
          <td>0.013760</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.076684</td>
          <td>0.542873</td>
          <td>25.982786</td>
          <td>0.160006</td>
          <td>25.344315</td>
          <td>0.171839</td>
          <td>24.193923</td>
          <td>0.142037</td>
          <td>0.157714</td>
          <td>0.144095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.600147</td>
          <td>2.413508</td>
          <td>28.326534</td>
          <td>0.672925</td>
          <td>27.221766</td>
          <td>0.266144</td>
          <td>26.035920</td>
          <td>0.157874</td>
          <td>26.088294</td>
          <td>0.301207</td>
          <td>25.802717</td>
          <td>0.494206</td>
          <td>0.073655</td>
          <td>0.051041</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.244003</td>
          <td>0.345017</td>
          <td>26.005780</td>
          <td>0.104852</td>
          <td>25.924656</td>
          <td>0.087688</td>
          <td>25.489869</td>
          <td>0.098034</td>
          <td>25.432666</td>
          <td>0.174521</td>
          <td>25.027985</td>
          <td>0.269400</td>
          <td>0.061370</td>
          <td>0.048975</td>
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
          <td>26.453675</td>
          <td>0.159631</td>
          <td>25.373769</td>
          <td>0.055938</td>
          <td>25.089189</td>
          <td>0.071596</td>
          <td>24.863307</td>
          <td>0.110837</td>
          <td>24.717373</td>
          <td>0.216139</td>
          <td>0.133861</td>
          <td>0.107340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.293614</td>
          <td>0.766198</td>
          <td>27.194703</td>
          <td>0.299512</td>
          <td>26.199447</td>
          <td>0.117564</td>
          <td>24.993180</td>
          <td>0.066846</td>
          <td>24.751883</td>
          <td>0.102137</td>
          <td>24.674642</td>
          <td>0.211769</td>
          <td>0.158380</td>
          <td>0.121546</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.949317</td>
          <td>0.245526</td>
          <td>26.212289</td>
          <td>0.119009</td>
          <td>26.746471</td>
          <td>0.299655</td>
          <td>25.563982</td>
          <td>0.205321</td>
          <td>25.277456</td>
          <td>0.346089</td>
          <td>0.148138</td>
          <td>0.135931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.001853</td>
          <td>0.283230</td>
          <td>26.153918</td>
          <td>0.118665</td>
          <td>26.044362</td>
          <td>0.096841</td>
          <td>26.158560</td>
          <td>0.173706</td>
          <td>25.961971</td>
          <td>0.269714</td>
          <td>25.624101</td>
          <td>0.428863</td>
          <td>0.042812</td>
          <td>0.029060</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.356670</td>
          <td>0.770703</td>
          <td>26.731356</td>
          <td>0.194029</td>
          <td>26.461779</td>
          <td>0.138838</td>
          <td>26.065673</td>
          <td>0.160011</td>
          <td>26.088843</td>
          <td>0.298079</td>
          <td>25.247705</td>
          <td>0.318950</td>
          <td>0.026892</td>
          <td>0.014157</td>
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
          <td>26.372335</td>
          <td>0.402956</td>
          <td>26.743982</td>
          <td>0.211757</td>
          <td>26.429448</td>
          <td>0.147200</td>
          <td>25.248846</td>
          <td>0.086083</td>
          <td>24.608131</td>
          <td>0.092447</td>
          <td>24.211878</td>
          <td>0.146796</td>
          <td>0.184964</td>
          <td>0.125776</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.775298</td>
          <td>0.925928</td>
          <td>27.604884</td>
          <td>0.347359</td>
          <td>26.604961</td>
          <td>0.134365</td>
          <td>26.616496</td>
          <td>0.217257</td>
          <td>25.869954</td>
          <td>0.214757</td>
          <td>25.203151</td>
          <td>0.264577</td>
          <td>0.020030</td>
          <td>0.013760</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.482750</td>
          <td>3.282042</td>
          <td>28.692675</td>
          <td>0.893869</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.897661</td>
          <td>0.149533</td>
          <td>25.176325</td>
          <td>0.149622</td>
          <td>24.484976</td>
          <td>0.183038</td>
          <td>0.157714</td>
          <td>0.144095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.056068</td>
          <td>0.507853</td>
          <td>27.158475</td>
          <td>0.224999</td>
          <td>26.148269</td>
          <td>0.153353</td>
          <td>25.402540</td>
          <td>0.151311</td>
          <td>24.733316</td>
          <td>0.187560</td>
          <td>0.073655</td>
          <td>0.051041</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.175303</td>
          <td>0.299163</td>
          <td>26.244270</td>
          <td>0.114873</td>
          <td>26.171038</td>
          <td>0.095421</td>
          <td>25.758431</td>
          <td>0.108163</td>
          <td>25.429839</td>
          <td>0.153200</td>
          <td>24.942318</td>
          <td>0.221007</td>
          <td>0.061370</td>
          <td>0.048975</td>
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
          <td>27.098660</td>
          <td>0.648596</td>
          <td>26.415379</td>
          <td>0.148741</td>
          <td>25.492142</td>
          <td>0.059484</td>
          <td>24.989799</td>
          <td>0.062687</td>
          <td>24.943845</td>
          <td>0.113934</td>
          <td>24.773322</td>
          <td>0.217172</td>
          <td>0.133861</td>
          <td>0.107340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.149235</td>
          <td>0.285821</td>
          <td>25.836423</td>
          <td>0.084508</td>
          <td>25.101455</td>
          <td>0.072626</td>
          <td>24.797690</td>
          <td>0.105018</td>
          <td>24.018855</td>
          <td>0.119505</td>
          <td>0.158380</td>
          <td>0.121546</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.807899</td>
          <td>0.544684</td>
          <td>26.453456</td>
          <td>0.160692</td>
          <td>26.405532</td>
          <td>0.139430</td>
          <td>26.025937</td>
          <td>0.163249</td>
          <td>26.745296</td>
          <td>0.518079</td>
          <td>26.119455</td>
          <td>0.642447</td>
          <td>0.148138</td>
          <td>0.135931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.154833</td>
          <td>0.290070</td>
          <td>26.385246</td>
          <td>0.127349</td>
          <td>26.133073</td>
          <td>0.090256</td>
          <td>25.983695</td>
          <td>0.128570</td>
          <td>25.863451</td>
          <td>0.216413</td>
          <td>25.255375</td>
          <td>0.279760</td>
          <td>0.042812</td>
          <td>0.029060</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.950202</td>
          <td>0.243693</td>
          <td>26.714862</td>
          <td>0.167361</td>
          <td>26.489678</td>
          <td>0.121861</td>
          <td>26.190227</td>
          <td>0.151802</td>
          <td>26.100699</td>
          <td>0.260425</td>
          <td>26.764767</td>
          <td>0.838504</td>
          <td>0.026892</td>
          <td>0.014157</td>
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
