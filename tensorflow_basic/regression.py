import tensorflow as tf
import os
import itertools
from pathlib import Path

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint', "../tmp/regression/model", 'Checkpoint directory.')
tf.flags.DEFINE_string('log', "../tmp/regression/log", 'Log directory.')
tf.flags.DEFINE_integer('num_steps', 1000, 'Number of steps to train.')
tf.flags.DEFINE_integer('save_steps', 100, 'Checkpoints save every save_steps.')


def find_last_step(path):
    """Return the last checkpoint step #.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def load_model(session, saver, checkpoint_dir):
    """Load model checkpoint
    """
    session.run(tf.global_variables_initializer())
    next_step = 1
    if tf.gfile.Exists(checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            next_step = find_last_step(ckpt.model_checkpoint_path) + 1
        else:
            for p in itertools.chain(Path(FLAGS.checkpoint).glob("checkpoint"),
                                     Path(FLAGS.checkpoint).glob("model*")):
                p.unlink()
    else:
        tf.gfile.MakeDirs(checkpoint_dir)
    return next_step


def save_model(session, saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(session, dir, global_step=step)


def main(_):
    # Linear model y = Wx + b
    W = tf.get_variable("W", shape=[1], initializer=tf.random_uniform_initializer(-1, 1))
    b = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(-1, 1))

    # Placeholder for input and prediction
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    y_pred = W * x + b

    loss = tf.reduce_sum(tf.square(y_pred - y))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = [1.0, 2.0, 3.0, 4.0]
    y_train = [1.5, 3.5, 5.5, 7.5]

    # Model checkpoint
    saver = tf.train.Saver(max_to_keep=3)

    # TensorBoard summary
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.log)

    # Profile info. or trainable parameters
    tf.profiler.Profiler().profile_name_scope(
        options=(tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()))

    with tf.Session() as sess:

        next_step = load_model(sess, saver, FLAGS.checkpoint)
        print(f"Restore checkpoint ({next_step!=0})")

        for i in range(next_step, next_step + FLAGS.num_steps):
            sess.run(train, {x: x_train, y: y_train})

            if i % FLAGS.save_steps == 1 or i == next_step:
                l_cost, summary = sess.run([loss, merged_summary], {x: x_train, y: y_train})
                print(f"i: {i} cost: {l_cost}")

                if i % FLAGS.save_steps == 1:
                    save_model(sess, saver, FLAGS.checkpoint, i)
                    summary_writer.add_summary(summary, i)
                    summary_writer.flush()

        l_W, l_b, l_cost = sess.run([W, b, loss], {x: x_train, y: y_train})
        print(f"W: {l_W} b: {l_b} cost: {l_cost}")
        # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
