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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fef947a8f70>



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
    0      23.994413  0.087387  0.049758  
    1      25.391064  0.066463  0.059403  
    2      24.304707  0.006130  0.005192  
    3      25.291103  0.123461  0.101668  
    4      25.096743  0.006248  0.006058  
    ...          ...       ...       ...  
    99995  24.737946  0.063336  0.040530  
    99996  24.224169  0.021391  0.013133  
    99997  25.613836  0.188249  0.115653  
    99998  25.274899  0.047278  0.030119  
    99999  25.699642  0.135129  0.075370  
    
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
          <td>27.696331</td>
          <td>0.879598</td>
          <td>26.917646</td>
          <td>0.197650</td>
          <td>26.233535</td>
          <td>0.096840</td>
          <td>25.320746</td>
          <td>0.070554</td>
          <td>24.875136</td>
          <td>0.090918</td>
          <td>24.120365</td>
          <td>0.105024</td>
          <td>0.087387</td>
          <td>0.049758</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.341961</td>
          <td>0.280634</td>
          <td>26.420598</td>
          <td>0.114048</td>
          <td>26.368000</td>
          <td>0.175553</td>
          <td>25.798515</td>
          <td>0.201525</td>
          <td>25.696533</td>
          <td>0.390347</td>
          <td>0.066463</td>
          <td>0.059403</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.941955</td>
          <td>0.526072</td>
          <td>28.998465</td>
          <td>0.929989</td>
          <td>27.560238</td>
          <td>0.297979</td>
          <td>26.180048</td>
          <td>0.149524</td>
          <td>24.961356</td>
          <td>0.098068</td>
          <td>24.370626</td>
          <td>0.130572</td>
          <td>0.006130</td>
          <td>0.005192</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.998853</td>
          <td>0.548252</td>
          <td>30.803939</td>
          <td>2.277343</td>
          <td>27.387579</td>
          <td>0.259007</td>
          <td>26.646919</td>
          <td>0.221948</td>
          <td>26.013405</td>
          <td>0.240997</td>
          <td>24.952729</td>
          <td>0.214307</td>
          <td>0.123461</td>
          <td>0.101668</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.192814</td>
          <td>0.295696</td>
          <td>26.179135</td>
          <td>0.104827</td>
          <td>25.952514</td>
          <td>0.075608</td>
          <td>25.616251</td>
          <td>0.091570</td>
          <td>25.729343</td>
          <td>0.190130</td>
          <td>24.830728</td>
          <td>0.193467</td>
          <td>0.006248</td>
          <td>0.006058</td>
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
          <td>26.713629</td>
          <td>0.444064</td>
          <td>26.440768</td>
          <td>0.131590</td>
          <td>25.385624</td>
          <td>0.045734</td>
          <td>25.098316</td>
          <td>0.057928</td>
          <td>24.772030</td>
          <td>0.083029</td>
          <td>24.861981</td>
          <td>0.198621</td>
          <td>0.063336</td>
          <td>0.040530</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.909111</td>
          <td>1.002776</td>
          <td>26.771204</td>
          <td>0.174656</td>
          <td>25.963216</td>
          <td>0.076326</td>
          <td>25.102902</td>
          <td>0.058164</td>
          <td>24.880289</td>
          <td>0.091331</td>
          <td>24.070293</td>
          <td>0.100521</td>
          <td>0.021391</td>
          <td>0.013133</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.632916</td>
          <td>0.417670</td>
          <td>26.639128</td>
          <td>0.156065</td>
          <td>26.326872</td>
          <td>0.105089</td>
          <td>26.147311</td>
          <td>0.145377</td>
          <td>26.159742</td>
          <td>0.271708</td>
          <td>32.342651</td>
          <td>5.741389</td>
          <td>0.188249</td>
          <td>0.115653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.849654</td>
          <td>0.491588</td>
          <td>26.112177</td>
          <td>0.098865</td>
          <td>26.128306</td>
          <td>0.088288</td>
          <td>26.031505</td>
          <td>0.131556</td>
          <td>25.753832</td>
          <td>0.194095</td>
          <td>24.979957</td>
          <td>0.219228</td>
          <td>0.047278</td>
          <td>0.030119</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.616433</td>
          <td>0.835990</td>
          <td>26.696278</td>
          <td>0.163871</td>
          <td>26.742885</td>
          <td>0.150717</td>
          <td>26.464029</td>
          <td>0.190415</td>
          <td>26.111931</td>
          <td>0.261313</td>
          <td>26.148477</td>
          <td>0.547576</td>
          <td>0.135129</td>
          <td>0.075370</td>
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
          <td>26.515281</td>
          <td>0.427496</td>
          <td>26.589037</td>
          <td>0.174344</td>
          <td>25.852340</td>
          <td>0.082814</td>
          <td>25.193131</td>
          <td>0.076010</td>
          <td>24.689389</td>
          <td>0.092313</td>
          <td>24.109672</td>
          <td>0.124913</td>
          <td>0.087387</td>
          <td>0.049758</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.601173</td>
          <td>0.396136</td>
          <td>26.492691</td>
          <td>0.144241</td>
          <td>26.729935</td>
          <td>0.281726</td>
          <td>25.807787</td>
          <td>0.239639</td>
          <td>25.179818</td>
          <td>0.305441</td>
          <td>0.066463</td>
          <td>0.059403</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.678463</td>
          <td>0.945269</td>
          <td>29.027324</td>
          <td>1.044181</td>
          <td>28.432801</td>
          <td>0.659169</td>
          <td>26.378763</td>
          <td>0.208234</td>
          <td>24.867203</td>
          <td>0.106086</td>
          <td>24.126081</td>
          <td>0.124568</td>
          <td>0.006130</td>
          <td>0.005192</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.486732</td>
          <td>0.857666</td>
          <td>27.898337</td>
          <td>0.506596</td>
          <td>27.430902</td>
          <td>0.323271</td>
          <td>26.462210</td>
          <td>0.232485</td>
          <td>25.850883</td>
          <td>0.254950</td>
          <td>26.784877</td>
          <td>0.980777</td>
          <td>0.123461</td>
          <td>0.101668</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.770803</td>
          <td>0.511816</td>
          <td>26.008491</td>
          <td>0.104136</td>
          <td>25.928011</td>
          <td>0.087046</td>
          <td>25.906381</td>
          <td>0.139360</td>
          <td>25.110650</td>
          <td>0.131098</td>
          <td>25.091414</td>
          <td>0.280900</td>
          <td>0.006248</td>
          <td>0.006058</td>
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
          <td>27.070169</td>
          <td>0.637673</td>
          <td>26.494312</td>
          <td>0.159808</td>
          <td>25.456520</td>
          <td>0.057920</td>
          <td>25.029066</td>
          <td>0.065241</td>
          <td>24.693696</td>
          <td>0.091986</td>
          <td>24.774179</td>
          <td>0.218369</td>
          <td>0.063336</td>
          <td>0.040530</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.117252</td>
          <td>0.267055</td>
          <td>25.985368</td>
          <td>0.091636</td>
          <td>25.093498</td>
          <td>0.068464</td>
          <td>24.845432</td>
          <td>0.104187</td>
          <td>24.037213</td>
          <td>0.115425</td>
          <td>0.021391</td>
          <td>0.013133</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.526523</td>
          <td>0.448491</td>
          <td>26.493210</td>
          <td>0.169258</td>
          <td>26.307373</td>
          <td>0.130686</td>
          <td>26.220557</td>
          <td>0.196441</td>
          <td>25.848298</td>
          <td>0.262581</td>
          <td>26.001393</td>
          <td>0.601309</td>
          <td>0.188249</td>
          <td>0.115653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.867311</td>
          <td>0.254001</td>
          <td>26.125252</td>
          <td>0.115827</td>
          <td>26.155590</td>
          <td>0.106826</td>
          <td>25.832406</td>
          <td>0.131430</td>
          <td>25.659732</td>
          <td>0.210301</td>
          <td>24.783953</td>
          <td>0.219255</td>
          <td>0.047278</td>
          <td>0.030119</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.853804</td>
          <td>0.557132</td>
          <td>26.870694</td>
          <td>0.225097</td>
          <td>26.860494</td>
          <td>0.202098</td>
          <td>26.473811</td>
          <td>0.234061</td>
          <td>25.893558</td>
          <td>0.263296</td>
          <td>25.929441</td>
          <td>0.553940</td>
          <td>0.135129</td>
          <td>0.075370</td>
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
          <td>27.608808</td>
          <td>0.859470</td>
          <td>26.538837</td>
          <td>0.151170</td>
          <td>26.139734</td>
          <td>0.095008</td>
          <td>25.244287</td>
          <td>0.070490</td>
          <td>24.605096</td>
          <td>0.076357</td>
          <td>23.998436</td>
          <td>0.100769</td>
          <td>0.087387</td>
          <td>0.049758</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.142040</td>
          <td>0.624939</td>
          <td>27.035865</td>
          <td>0.227610</td>
          <td>26.528639</td>
          <td>0.131760</td>
          <td>26.529493</td>
          <td>0.211740</td>
          <td>25.661561</td>
          <td>0.188589</td>
          <td>25.849383</td>
          <td>0.459212</td>
          <td>0.066463</td>
          <td>0.059403</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.267582</td>
          <td>0.514294</td>
          <td>26.019842</td>
          <td>0.130295</td>
          <td>25.055421</td>
          <td>0.106531</td>
          <td>24.246442</td>
          <td>0.117286</td>
          <td>0.006130</td>
          <td>0.005192</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.233282</td>
          <td>1.292577</td>
          <td>30.373231</td>
          <td>2.033842</td>
          <td>27.360457</td>
          <td>0.290433</td>
          <td>26.126297</td>
          <td>0.165907</td>
          <td>25.470018</td>
          <td>0.175974</td>
          <td>25.215763</td>
          <td>0.306446</td>
          <td>0.123461</td>
          <td>0.101668</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.601479</td>
          <td>0.407871</td>
          <td>26.063493</td>
          <td>0.094779</td>
          <td>26.059239</td>
          <td>0.083120</td>
          <td>25.726252</td>
          <td>0.100904</td>
          <td>25.414276</td>
          <td>0.145437</td>
          <td>25.387435</td>
          <td>0.306115</td>
          <td>0.006248</td>
          <td>0.006058</td>
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
          <td>26.022807</td>
          <td>0.263852</td>
          <td>26.374665</td>
          <td>0.128262</td>
          <td>25.480771</td>
          <td>0.051653</td>
          <td>25.122832</td>
          <td>0.061554</td>
          <td>24.890669</td>
          <td>0.095614</td>
          <td>25.029834</td>
          <td>0.236864</td>
          <td>0.063336</td>
          <td>0.040530</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.235896</td>
          <td>1.213466</td>
          <td>26.905166</td>
          <td>0.196281</td>
          <td>25.878337</td>
          <td>0.071112</td>
          <td>25.227326</td>
          <td>0.065245</td>
          <td>24.872029</td>
          <td>0.091058</td>
          <td>24.195947</td>
          <td>0.112683</td>
          <td>0.021391</td>
          <td>0.013133</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.466849</td>
          <td>0.431039</td>
          <td>26.779536</td>
          <td>0.216746</td>
          <td>26.525014</td>
          <td>0.158589</td>
          <td>27.078799</td>
          <td>0.395737</td>
          <td>26.375838</td>
          <td>0.401610</td>
          <td>25.580221</td>
          <td>0.444194</td>
          <td>0.188249</td>
          <td>0.115653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.466678</td>
          <td>0.372135</td>
          <td>26.312953</td>
          <td>0.119921</td>
          <td>26.252540</td>
          <td>0.100535</td>
          <td>25.803349</td>
          <td>0.110254</td>
          <td>25.888294</td>
          <td>0.221567</td>
          <td>24.880920</td>
          <td>0.206012</td>
          <td>0.047278</td>
          <td>0.030119</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.237285</td>
          <td>0.335109</td>
          <td>26.610916</td>
          <td>0.171207</td>
          <td>26.552671</td>
          <td>0.146168</td>
          <td>26.962292</td>
          <td>0.327103</td>
          <td>25.940535</td>
          <td>0.257678</td>
          <td>27.401734</td>
          <td>1.331527</td>
          <td>0.135129</td>
          <td>0.075370</td>
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
