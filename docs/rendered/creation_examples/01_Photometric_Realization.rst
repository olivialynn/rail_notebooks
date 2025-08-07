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

    <pzflow.flow.Flow at 0x7f339064aa70>



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
    0      23.994413  0.054549  0.043334  
    1      25.391064  0.020580  0.019571  
    2      24.304707  0.033906  0.018646  
    3      25.291103  0.026349  0.021277  
    4      25.096743  0.031637  0.017439  
    ...          ...       ...       ...  
    99995  24.737946  0.074344  0.064091  
    99996  24.224169  0.016855  0.011442  
    99997  25.613836  0.039592  0.022633  
    99998  25.274899  0.017052  0.009968  
    99999  25.699642  0.033113  0.026356  
    
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
          <td>27.375854</td>
          <td>0.713529</td>
          <td>26.680959</td>
          <td>0.161744</td>
          <td>26.060804</td>
          <td>0.083192</td>
          <td>25.158744</td>
          <td>0.061119</td>
          <td>24.796485</td>
          <td>0.084838</td>
          <td>24.161704</td>
          <td>0.108887</td>
          <td>0.054549</td>
          <td>0.043334</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.978130</td>
          <td>0.540092</td>
          <td>28.541361</td>
          <td>0.690885</td>
          <td>26.512300</td>
          <td>0.123516</td>
          <td>25.863775</td>
          <td>0.113726</td>
          <td>26.124023</td>
          <td>0.263908</td>
          <td>25.917590</td>
          <td>0.461931</td>
          <td>0.020580</td>
          <td>0.019571</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.818097</td>
          <td>0.365628</td>
          <td>26.116192</td>
          <td>0.141535</td>
          <td>25.016174</td>
          <td>0.102891</td>
          <td>24.261912</td>
          <td>0.118821</td>
          <td>0.033906</td>
          <td>0.018646</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.396382</td>
          <td>0.625062</td>
          <td>27.770723</td>
          <td>0.352301</td>
          <td>26.184077</td>
          <td>0.150042</td>
          <td>25.676426</td>
          <td>0.181815</td>
          <td>25.198100</td>
          <td>0.262473</td>
          <td>0.026349</td>
          <td>0.021277</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.515510</td>
          <td>0.381593</td>
          <td>26.096157</td>
          <td>0.097488</td>
          <td>25.945695</td>
          <td>0.075154</td>
          <td>25.605924</td>
          <td>0.090743</td>
          <td>25.535223</td>
          <td>0.161242</td>
          <td>25.241073</td>
          <td>0.271836</td>
          <td>0.031637</td>
          <td>0.017439</td>
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
          <td>29.845051</td>
          <td>2.503017</td>
          <td>26.415550</td>
          <td>0.128751</td>
          <td>25.432554</td>
          <td>0.047680</td>
          <td>25.034573</td>
          <td>0.054741</td>
          <td>24.800971</td>
          <td>0.085174</td>
          <td>24.913746</td>
          <td>0.207436</td>
          <td>0.074344</td>
          <td>0.064091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.866750</td>
          <td>0.497837</td>
          <td>26.694616</td>
          <td>0.163639</td>
          <td>26.078152</td>
          <td>0.084473</td>
          <td>25.193799</td>
          <td>0.063049</td>
          <td>25.008499</td>
          <td>0.102203</td>
          <td>24.154084</td>
          <td>0.108165</td>
          <td>0.016855</td>
          <td>0.011442</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.130037</td>
          <td>0.602119</td>
          <td>27.101302</td>
          <td>0.230388</td>
          <td>26.449729</td>
          <td>0.116979</td>
          <td>26.331054</td>
          <td>0.170126</td>
          <td>26.769780</td>
          <td>0.439171</td>
          <td>25.770260</td>
          <td>0.413131</td>
          <td>0.039592</td>
          <td>0.022633</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.098004</td>
          <td>0.273883</td>
          <td>26.509867</td>
          <td>0.139674</td>
          <td>26.252360</td>
          <td>0.098452</td>
          <td>25.902664</td>
          <td>0.117643</td>
          <td>26.243762</td>
          <td>0.290860</td>
          <td>25.068495</td>
          <td>0.235943</td>
          <td>0.017052</td>
          <td>0.009968</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.062779</td>
          <td>0.574022</td>
          <td>26.828480</td>
          <td>0.183339</td>
          <td>26.435730</td>
          <td>0.115562</td>
          <td>26.248511</td>
          <td>0.158558</td>
          <td>25.687322</td>
          <td>0.183499</td>
          <td>25.774339</td>
          <td>0.414423</td>
          <td>0.033113</td>
          <td>0.026356</td>
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
          <td>28.055974</td>
          <td>1.184849</td>
          <td>27.036668</td>
          <td>0.251519</td>
          <td>25.987144</td>
          <td>0.092436</td>
          <td>25.246921</td>
          <td>0.078991</td>
          <td>24.710313</td>
          <td>0.093214</td>
          <td>23.795515</td>
          <td>0.094124</td>
          <td>0.054549</td>
          <td>0.043334</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.113046</td>
          <td>0.653712</td>
          <td>27.159104</td>
          <td>0.276374</td>
          <td>26.754670</td>
          <td>0.178326</td>
          <td>26.118911</td>
          <td>0.167413</td>
          <td>25.511949</td>
          <td>0.185031</td>
          <td>25.083797</td>
          <td>0.279500</td>
          <td>0.020580</td>
          <td>0.019571</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.153179</td>
          <td>1.247008</td>
          <td>30.903806</td>
          <td>2.512979</td>
          <td>28.856037</td>
          <td>0.873813</td>
          <td>25.989449</td>
          <td>0.150048</td>
          <td>24.866778</td>
          <td>0.106304</td>
          <td>24.443363</td>
          <td>0.164070</td>
          <td>0.033906</td>
          <td>0.018646</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.192471</td>
          <td>1.150341</td>
          <td>27.527540</td>
          <td>0.336747</td>
          <td>26.595866</td>
          <td>0.249772</td>
          <td>25.459844</td>
          <td>0.177144</td>
          <td>26.004159</td>
          <td>0.566741</td>
          <td>0.026349</td>
          <td>0.021277</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.451577</td>
          <td>0.403246</td>
          <td>26.118383</td>
          <td>0.114818</td>
          <td>26.075308</td>
          <td>0.099273</td>
          <td>25.497264</td>
          <td>0.097842</td>
          <td>25.390685</td>
          <td>0.167074</td>
          <td>24.646954</td>
          <td>0.194901</td>
          <td>0.031637</td>
          <td>0.017439</td>
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
          <td>27.246687</td>
          <td>0.722351</td>
          <td>26.370199</td>
          <td>0.144531</td>
          <td>25.386993</td>
          <td>0.054822</td>
          <td>25.052004</td>
          <td>0.067042</td>
          <td>25.033770</td>
          <td>0.124605</td>
          <td>24.541543</td>
          <td>0.180774</td>
          <td>0.074344</td>
          <td>0.064091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.664350</td>
          <td>2.461018</td>
          <td>26.633321</td>
          <td>0.178482</td>
          <td>25.990055</td>
          <td>0.091980</td>
          <td>25.193503</td>
          <td>0.074766</td>
          <td>24.844715</td>
          <td>0.104083</td>
          <td>24.486187</td>
          <td>0.169854</td>
          <td>0.016855</td>
          <td>0.011442</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.828347</td>
          <td>0.210821</td>
          <td>26.296851</td>
          <td>0.120601</td>
          <td>26.359700</td>
          <td>0.205629</td>
          <td>25.730134</td>
          <td>0.222637</td>
          <td>27.703473</td>
          <td>1.579354</td>
          <td>0.039592</td>
          <td>0.022633</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.681700</td>
          <td>0.479384</td>
          <td>26.290104</td>
          <td>0.133058</td>
          <td>26.363707</td>
          <td>0.127442</td>
          <td>26.210333</td>
          <td>0.180802</td>
          <td>25.855790</td>
          <td>0.246363</td>
          <td>25.096105</td>
          <td>0.282114</td>
          <td>0.017052</td>
          <td>0.009968</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.585018</td>
          <td>0.893380</td>
          <td>26.547862</td>
          <td>0.166326</td>
          <td>26.564038</td>
          <td>0.151817</td>
          <td>26.129713</td>
          <td>0.169243</td>
          <td>25.687793</td>
          <td>0.214823</td>
          <td>27.547957</td>
          <td>1.461782</td>
          <td>0.033113</td>
          <td>0.026356</td>
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
          <td>26.948086</td>
          <td>0.208198</td>
          <td>26.123739</td>
          <td>0.090786</td>
          <td>25.166832</td>
          <td>0.063674</td>
          <td>24.753843</td>
          <td>0.084362</td>
          <td>23.882165</td>
          <td>0.088083</td>
          <td>0.054549</td>
          <td>0.043334</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.442938</td>
          <td>2.152115</td>
          <td>28.083427</td>
          <td>0.501059</td>
          <td>26.425323</td>
          <td>0.115140</td>
          <td>26.178855</td>
          <td>0.150209</td>
          <td>25.920651</td>
          <td>0.224328</td>
          <td>25.137875</td>
          <td>0.251158</td>
          <td>0.020580</td>
          <td>0.019571</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.235926</td>
          <td>0.651993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.320443</td>
          <td>0.538792</td>
          <td>25.809274</td>
          <td>0.109587</td>
          <td>25.154790</td>
          <td>0.117285</td>
          <td>24.289225</td>
          <td>0.122937</td>
          <td>0.033906</td>
          <td>0.018646</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.506122</td>
          <td>0.677978</td>
          <td>27.474595</td>
          <td>0.280040</td>
          <td>26.368024</td>
          <td>0.176939</td>
          <td>25.275837</td>
          <td>0.129980</td>
          <td>24.763875</td>
          <td>0.184269</td>
          <td>0.026349</td>
          <td>0.021277</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.463518</td>
          <td>0.368455</td>
          <td>26.069393</td>
          <td>0.095955</td>
          <td>26.046965</td>
          <td>0.082910</td>
          <td>25.732472</td>
          <td>0.102335</td>
          <td>25.489940</td>
          <td>0.156446</td>
          <td>25.323076</td>
          <td>0.292963</td>
          <td>0.031637</td>
          <td>0.017439</td>
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
          <td>26.367933</td>
          <td>0.353417</td>
          <td>26.491462</td>
          <td>0.144891</td>
          <td>25.405813</td>
          <td>0.049545</td>
          <td>25.087728</td>
          <td>0.061238</td>
          <td>24.750198</td>
          <td>0.086612</td>
          <td>25.137617</td>
          <td>0.265023</td>
          <td>0.074344</td>
          <td>0.064091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.711244</td>
          <td>0.444007</td>
          <td>27.153433</td>
          <td>0.241085</td>
          <td>25.910163</td>
          <td>0.073036</td>
          <td>25.131646</td>
          <td>0.059846</td>
          <td>24.938094</td>
          <td>0.096357</td>
          <td>24.206027</td>
          <td>0.113508</td>
          <td>0.016855</td>
          <td>0.011442</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.678361</td>
          <td>1.535735</td>
          <td>26.776102</td>
          <td>0.177438</td>
          <td>26.550055</td>
          <td>0.129386</td>
          <td>26.492588</td>
          <td>0.197792</td>
          <td>26.400705</td>
          <td>0.333952</td>
          <td>24.973563</td>
          <td>0.221064</td>
          <td>0.039592</td>
          <td>0.022633</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.167031</td>
          <td>0.290105</td>
          <td>26.232417</td>
          <td>0.110067</td>
          <td>26.214920</td>
          <td>0.095523</td>
          <td>25.958474</td>
          <td>0.123828</td>
          <td>25.831734</td>
          <td>0.207738</td>
          <td>25.702812</td>
          <td>0.393193</td>
          <td>0.017052</td>
          <td>0.009968</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.288612</td>
          <td>0.321685</td>
          <td>26.711129</td>
          <td>0.167646</td>
          <td>26.788270</td>
          <td>0.158539</td>
          <td>26.475765</td>
          <td>0.194648</td>
          <td>25.802436</td>
          <td>0.204519</td>
          <td>25.371497</td>
          <td>0.305540</td>
          <td>0.033113</td>
          <td>0.026356</td>
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
