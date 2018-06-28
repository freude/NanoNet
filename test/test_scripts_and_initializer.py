import unittest
import tb.tb_script as tb_script
import tb.tbmpi_script as tbmpi_script


class Test(unittest.TestCase):

    def setUp(self):
        self.parser = tb_script.create_parser()

    def test_tb_without_mpi_sense(self):

        args = self.parser.parse_args(['./input_samples/input.yaml', '-S=1'])
        ans = tb_script.main1(args.param_file, args.k_points_file, args.xyz, args.show, args.save)
        self.assertEquals(ans, 0)

    def test_tb_without_mpi_sparse(self):
        args = self.parser.parse_args(['./input_samples/input.yaml', '-S=1'])
        ans = tbmpi_script.main1(args.param_file, args.k_points_file, args.xyz, args.show, args.save)
        self.assertEquals(ans, 0)


if __name__ == '__main__':

    unittest.main()
